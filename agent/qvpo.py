import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agent.model import Critic, CriticQR
from agent.diffusion import Diffusion
from agent.helpers import quantile_huber_loss
import os


class QVPO(object):
    def __init__(self,
                 args,
                 state_dim,
                 action_space,
                 memory,
                 device,
                 ):
        action_dim = int(np.prod(action_space.shape))

        self.policy_type = args.policy_type
        if self.policy_type == 'Diffusion':
            self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, noise_ratio=args.noise_ratio,
                                   beta_schedule=args.beta_schedule, n_timesteps=args.n_timesteps, behavior_sample=args.behavior_sample,
                                   eval_sample=args.eval_sample, deterministic=args.deterministic).to(device)
            sampling_precision = getattr(args, 'sampling_precision', 'float16')
            self.actor.sampling_dtype = (
                torch.float16 if sampling_precision == 'float16' else None
            )
            self.running_q_std = torch.tensor(1.0, device=device)
            self.running_q_mean = torch.tensor(0.0, device=device)
            self.beta = args.beta
            self.alpha_mean = args.alpha_mean
            self.alpha_std = args.alpha_std
            self.chosen = args.chosen
            self.q_neg = args.q_neg

            self.weighted = args.weighted
            self.train_sample = args.train_sample

            self.q_transform = args.q_transform
            self.policy_freq = args.policy_freq

            self.cut = args.cut

            # Keep the scheduled coefficient in device storage. Passing this
            # tensor into the actor graph avoids recompiling whenever main.py
            # updates the Python-facing value.
            self._entropy_alpha = float(args.entropy_alpha)
            self._entropy_alpha_tensor = torch.tensor(
                self._entropy_alpha, device=device
            )

        else:
            raise NotImplementedError

        fused_adam = device.type == 'cuda'
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=args.diffusion_lr, eps=1e-5,
            fused=fused_adam
        )

        self.memory = memory
        self.ac_grad_norm = args.ac_grad_norm

        self.step = 0
        self.tau = args.tau
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.behavior_sample = args.target_sample
        self.update_actor_target_every = args.update_actor_target_every

        self.critic_type = getattr(args, 'critic_type', 'quantile')
        self.n_quantiles = getattr(args, 'n_quantiles', 25)
        self.n_nets = getattr(args, 'n_nets', 5)
        self.top_quantiles_to_drop_per_net = getattr(
            args, 'top_quantiles_to_drop_per_net', 2
        )
        self.quantiles_total = self.n_quantiles * self.n_nets
        kept_quantiles = self.quantiles_total - (
            self.top_quantiles_to_drop_per_net * self.n_nets
        )
        if kept_quantiles <= 0:
            raise ValueError('top_quantiles_to_drop_per_net removes every quantile')

        if self.critic_type == 'quantile':
            self.critic = CriticQR(
                state_dim, action_dim, n_quantiles=self.n_quantiles,
                n_nets=self.n_nets
            ).to(device)
        elif self.critic_type == 'double':
            self.critic = Critic(state_dim, action_dim).to(device)
        else:
            raise ValueError("critic_type must be 'quantile' or 'double'")
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=args.critic_lr, eps=1e-5,
            fused=fused_adam
        )

        self.compile_enabled = getattr(args, 'compile', False)
        self.compile_mode = getattr(args, 'compile_mode', 'reduce-overhead')
        self.compile_backend = getattr(args, 'compile_backend', 'inductor')
        self._configure_compute_graphs()

        self.action_dim = action_dim

        self.device = device

        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = (action_space.high - action_space.low) / 2.
            self.action_bias = (action_space.high + action_space.low) / 2.

    def _configure_compute_graphs(self):
        """Build full-graph entry points for all hot tensor work.

        Each training path owns its complete fixed-step diffusion loop so that
        Dynamo/Inductor can see sampling, candidate ranking, and loss creation
        as one static graph instead of dispatching one graph per DDPM step.
        """
        critic_forward = self.critic.functional_forward
        train_sample = self.train_sample
        target_sample = self.actor_target.behavior_sample
        chosen = self.chosen
        action_dim = self.actor.action_dim
        use_entropy_augmentation = self._entropy_alpha > 0.0

        def critic_value(state, action, parameters):
            output = critic_forward(state, action, parameters)
            if self.critic_type == 'quantile':
                return output.mean(dim=(1, 2)).unsqueeze(-1)
            return torch.minimum(output[0], output[1])

        if self.critic_type == 'quantile':
            keep = self.quantiles_total - (
                self.top_quantiles_to_drop_per_net * self.n_nets
            )

            def critic_train_loss(
                    states, actions, rewards, next_states, masks, parameters,
                    target_parameters, actor_target_parameters, noise_ratio):
                current_z = critic_forward(states, actions, parameters)
                raw_batch_size = next_states.shape[0]
                expanded_next_states = next_states.repeat(target_sample, 1)
                initial_noise = torch.randn(
                    (expanded_next_states.shape[0], action_dim),
                    device=next_states.device, dtype=next_states.dtype,
                )
                candidate_next_actions = self.actor_target.functional_sample_loop(
                    expanded_next_states, initial_noise,
                    actor_target_parameters, noise_ratio,
                ).clamp(-1.0, 1.0).detach()
                candidate_z = critic_forward(
                    expanded_next_states, candidate_next_actions,
                    target_parameters
                )
                candidate_values = candidate_z.mean(dim=(1, 2)).view(
                    target_sample, raw_batch_size
                ).transpose(0, 1)
                best_indices = torch.argmax(candidate_values, dim=1)
                candidate_z = candidate_z.view(
                    target_sample, raw_batch_size,
                    self.n_nets, self.n_quantiles
                ).permute(1, 0, 2, 3)
                selected_z = candidate_z.gather(
                    1,
                    best_indices[:, None, None, None].expand(
                        -1, 1, self.n_nets, self.n_quantiles
                    ),
                ).squeeze(1)
                sorted_z = torch.sort(
                    selected_z.flatten(1), dim=1
                ).values[:, :keep]
                target = rewards + masks * sorted_z
                return quantile_huber_loss(current_z, target.detach())
        else:
            def critic_train_loss(
                    states, actions, rewards, next_states, masks, parameters,
                    target_parameters, actor_target_parameters, noise_ratio):
                current_q1, current_q2 = critic_forward(
                    states, actions, parameters
                )
                raw_batch_size = next_states.shape[0]
                expanded_next_states = next_states.repeat(target_sample, 1)
                initial_noise = torch.randn(
                    (expanded_next_states.shape[0], action_dim),
                    device=next_states.device, dtype=next_states.dtype,
                )
                candidate_next_actions = self.actor_target.functional_sample_loop(
                    expanded_next_states, initial_noise,
                    actor_target_parameters, noise_ratio,
                ).clamp(-1.0, 1.0).detach()
                candidate_q1, candidate_q2 = critic_forward(
                    expanded_next_states, candidate_next_actions,
                    target_parameters
                )
                candidate_q = torch.minimum(
                    candidate_q1, candidate_q2
                ).view(target_sample, raw_batch_size, 1).transpose(0, 1)
                best_indices = torch.argmax(candidate_q, dim=1, keepdim=True)
                selected_q = candidate_q.gather(1, best_indices).squeeze(1)
                target_q = rewards + masks * selected_q
                target_q = target_q.detach()
                return (F.mse_loss(current_q1, target_q) +
                        F.mse_loss(current_q2, target_q))

        self._critic_value_graph = critic_value
        self._critic_train_loss_graph = critic_train_loss

        def actor_train_loss(
                states, critic_parameters, actor_parameters, noise_ratio,
                running_q_mean, running_q_std, entropy_alpha):
            """Sample/rank detached candidates and build actor loss."""
            raw_batch_size = states.shape[0]
            expanded_states = states.repeat(train_sample, 1)
            initial_noise = torch.randn(
                (expanded_states.shape[0], action_dim),
                device=states.device, dtype=states.dtype,
            )
            candidate_actions = self.actor.functional_sample_loop(
                expanded_states, initial_noise, actor_parameters, noise_ratio,
            ).clamp(-1.0, 1.0).detach()
            q = critic_forward(
                expanded_states, candidate_actions, critic_parameters
            )
            if self.critic_type == 'quantile':
                q = q.mean(dim=(1, 2)).unsqueeze(-1)
            else:
                q = torch.minimum(q[0], q[1])
            q = q.detach()
            actions = candidate_actions.view(
                train_sample, raw_batch_size, action_dim
            ).transpose(0, 1)
            q = q.view(train_sample, raw_batch_size, 1).transpose(0, 1)
            mean = q.mean()
            std = q.std()
            v = q.mean(dim=1, keepdim=True)
            if chosen == 1:
                selected_q, indices = torch.max(q, dim=1, keepdim=True)
            else:
                selected_q, indices = torch.topk(q, k=chosen, dim=1)
            selected_actions = actions.gather(
                1, indices.expand(-1, -1, action_dim)
            ).reshape(raw_batch_size * chosen, action_dim)
            selected_states = states[:, None, :].expand(
                -1, chosen, -1
            ).reshape(raw_batch_size * chosen, -1)
            selected_q = selected_q.reshape(raw_batch_size * chosen, 1)

            next_running_q_mean = running_q_mean
            next_running_q_std = running_q_std
            if self.weighted:
                next_running_q_std = (
                    running_q_std + self.alpha_std * (std - running_q_std)
                )
                next_running_q_mean = (
                    running_q_mean
                    + self.alpha_mean * (mean - running_q_mean)
                )
                if self.q_transform == 'qadv':
                    weights = (
                        selected_q.view(raw_batch_size, chosen, 1) - v
                    ).clamp_min(0.0).reshape(raw_batch_size * chosen, 1)
                elif self.q_transform == 'qrelu':
                    weights = selected_q.clamp_min(-self.q_neg) + self.q_neg
                elif self.q_transform == 'qexpn':
                    weights = torch.exp(
                        self.beta * ((selected_q - next_running_q_mean)
                                     / next_running_q_std).clamp(-8.0, 8.0)
                    )
                elif self.q_transform == 'qcut':
                    weights = selected_q.clamp_min(self.cut)
                elif self.q_transform == 'qcut0n':
                    weights = (
                        self.beta * (selected_q - next_running_q_mean)
                    ).clamp_min(0.0)
                elif self.q_transform == 'qcut1n':
                    weights = (
                        self.beta * (selected_q - next_running_q_mean)
                    ).clamp_min(1.0)
                else:
                    raise ValueError(f'unsupported q_transform: {self.q_transform}')

                if use_entropy_augmentation:
                    entropy_samples = 10
                    rand_states = selected_states.unsqueeze(0).expand(
                        entropy_samples, -1, -1
                    ).reshape(raw_batch_size * chosen * entropy_samples, -1)
                    rand_policy_actions = torch.empty(
                        raw_batch_size * chosen * entropy_samples,
                        action_dim,
                        device=states.device,
                    ).uniform_(-1.0, 1.0)
                    rand_weights = weights.unsqueeze(0).expand(
                        entropy_samples, -1, -1
                    ).reshape(
                        raw_batch_size * chosen * entropy_samples, 1
                    ) * entropy_alpha
                    selected_actions = torch.cat(
                        (selected_actions, rand_policy_actions), dim=0
                    )
                    selected_states = torch.cat(
                        (selected_states, rand_states), dim=0
                    )
                    weights = torch.cat((weights, rand_weights), dim=0)
                loss = self.actor.loss(
                    selected_actions.detach(), selected_states,
                    weights=weights.detach()
                )
            else:
                loss = self.actor.loss(
                    selected_actions.detach(), selected_states
                )
            return loss, next_running_q_mean, next_running_q_std

        self._actor_train_loss_graph = actor_train_loss

        def interaction_action(state, initial_noise, actor_parameters,
                               critic_parameters, noise_ratio):
            """Generate, rank, and select online actions in one tensor graph."""
            raw_batch_size = state.shape[0]
            sample_count = initial_noise.shape[0] // raw_batch_size
            expanded_states = state.repeat(sample_count, 1)
            actions = self.actor.functional_sample_loop(
                expanded_states, initial_noise, actor_parameters, noise_ratio
            ).clamp(-1.0, 1.0)
            q = critic_forward(expanded_states, actions, critic_parameters)
            if self.critic_type == 'quantile':
                q = q.mean(dim=(1, 2)).unsqueeze(-1)
            else:
                q = torch.minimum(q[0], q[1])
            actions = actions.view(
                sample_count, raw_batch_size, action_dim
            ).transpose(0, 1)
            q = q.view(sample_count, raw_batch_size, 1).transpose(0, 1)
            indices = torch.argmax(q, dim=1, keepdim=True)
            return actions.gather(
                1, indices.expand(-1, -1, action_dim)
            ).reshape(raw_batch_size, action_dim)

        self._interaction_action_graph = interaction_action

        if self.compile_enabled:
            compile_options = dict(fullgraph=True, dynamic=False,
                                   backend=self.compile_backend)
            if self.compile_backend == 'inductor':
                compile_options['mode'] = self.compile_mode
            self._critic_value_graph = torch.compile(
                self._critic_value_graph, **compile_options
            )
            self._critic_train_loss_graph = torch.compile(
                self._critic_train_loss_graph, **compile_options
            )
            self._actor_train_loss_graph = torch.compile(
                self._actor_train_loss_graph, **compile_options
            )
            self._interaction_action_graph = torch.compile(
                self._interaction_action_graph, **compile_options
            )

    def _critic_value(self, critic, state, action):
        return self._critic_value_graph(
            state, action, tuple(critic.parameters())
        )

    @property
    def entropy_alpha(self):
        return self._entropy_alpha

    @entropy_alpha.setter
    def entropy_alpha(self, value):
        self._entropy_alpha = float(value)
        self._entropy_alpha_tensor.fill_(self._entropy_alpha)

    def _online_q(self, state, action):
        return self._critic_value(self.critic, state, action)

    def append_memory(self, state, action, reward, next_state, mask):
        action = (action - self.action_bias) / self.action_scale

        self.memory.append(state, action, reward, next_state, mask)

    @torch.no_grad()
    def sample_action(self, state, eval=False):
        state = torch.as_tensor(
            state.reshape(1, -1), dtype=torch.float32, device=self.device
        )
        sample_count = (
            self.actor.eval_sample if eval else self.actor.behavior_sample
        )
        initial_noise = torch.randn(
            (state.shape[0] * sample_count, self.action_dim),
            device=self.device,
        )
        noise_ratio = (
            0.0 if self.actor.deterministic and eval
            else self.actor.max_noise_ratio
        )
        action = self._interaction_action_graph(
            state,
            initial_noise,
            tuple(self.actor.model.parameters()),
            tuple(self.critic.parameters()),
            noise_ratio,
        ).cpu().numpy().flatten()
        action = action.clip(-1, 1)
        action = action * self.action_scale + self.action_bias
        return action

    def train(self, t, iterations, batch_size=256, log_writer=None):
        for itr in range(iterations):
            # Sample replay buffer / batch
            states, actions, rewards, next_states, masks = self.memory.sample(batch_size)

            """ Q Training """
            critic_loss = self._critic_train_loss_graph(
                states, actions, rewards, next_states,
                masks,
                tuple(self.critic.parameters()),
                tuple(self.critic_target.parameters()),
                tuple(self.actor_target.model.parameters()),
                self.actor_target.noise_ratio,
            )

            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            if self.ac_grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.ac_grad_norm,
                                                             norm_type=2, foreach=True)
                # if self.step % 10 == 0:
                #     log_writer.add_scalar('Critic Grad Norm', critic_grad_norms.max().item(), self.step)
            self.critic_optimizer.step()

            """ Policy Training """
            if t % self.policy_freq == 0:
                actor_states, _, _, _, _ = self.memory.sample(batch_size)
                actor_loss, next_running_q_mean, next_running_q_std = (
                    self._actor_train_loss_graph(
                        actor_states,
                        tuple(self.critic.parameters()),
                        tuple(self.actor.model.parameters()),
                        self.actor.noise_ratio,
                        self.running_q_mean,
                        self.running_q_std,
                        self._entropy_alpha_tensor,
                    )
                )
                self.running_q_mean = next_running_q_mean.detach()
                self.running_q_std = next_running_q_std.detach()

                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                if self.ac_grad_norm > 0:
                    actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.ac_grad_norm,
                                                                norm_type=2, foreach=True)
                    # if self.step % 10 == 0:
                    #     log_writer.add_scalar('Actor Grad Norm', actor_grad_norms.max().item(), self.step)
                self.actor_optimizer.step()

            """ Step Target network """
            with torch.no_grad():
                torch._foreach_lerp_(
                    list(self.critic_target.parameters()),
                    list(self.critic.parameters()), self.tau
                )

            if self.step % self.update_actor_target_every == 0:
                with torch.no_grad():
                    torch._foreach_lerp_(
                        list(self.actor_target.parameters()),
                        list(self.actor.parameters()), self.tau
                    )

            self.step += 1

    def save_model(self, dir, id=None):
        os.makedirs(dir, exist_ok=True)
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic_{id}.pth')

        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic.pth')


    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth', map_location=self.device))
            self.critic.load_state_dict(torch.load(f'{dir}/critic_{id}.pth', map_location=self.device))

        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth', map_location=self.device))
            self.critic.load_state_dict(torch.load(f'{dir}/critic.pth', map_location=self.device))


    def get_policy(self, states, times):
        batch_size = states.shape[0]
        states = states.unsqueeze(1).repeat(1, times, 1).view(times*batch_size, -1)
        actions = self.actor(states, eval=False, normal=True)
        return actions

    def get_value(self, states, actions):
        action_shape = actions.shape[0]
        state_shape = states.shape[0]
        rep = int(action_shape / state_shape)
        states = states.unsqueeze(1).repeat(1, rep, 1).view(rep*state_shape, -1)
        if self.critic_type == 'quantile':
            q = self._online_q(states, actions).view(state_shape, rep, 1)
            return q, q
        q1, q2 = self.critic(states, actions)
        return q1.view(state_shape, rep, 1), q2.view(state_shape, rep, 1)
