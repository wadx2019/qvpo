import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from agent.model import MLP, Critic, CriticQR
from agent.diffusion import Diffusion
from agent.vae import VAE
from agent.helpers import EMA, quantile_huber_loss
from agent.q_transform import *
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
            self.running_q_std = 1.0
            self.running_q_mean = 0.0
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
            self.epsilon = args.epsilon

            self.entropy_alpha = args.entropy_alpha

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

        Actor, critic, and denoiser tensor work use reusable full graphs.
        """
        critic_forward = self.critic.functional_forward

        def critic_value(state, action, parameters):
            output = critic_forward(state, action, parameters)
            if self.critic_type == 'quantile':
                return output.mean(dim=(1, 2)).unsqueeze(-1)
            return torch.minimum(output[0], output[1])

        if self.critic_type == 'quantile':
            keep = self.quantiles_total - (
                self.top_quantiles_to_drop_per_net * self.n_nets
            )

            def critic_loss(states, actions, rewards, next_states,
                            next_actions, masks, parameters, target_parameters):
                current_z = critic_forward(states, actions, parameters)
                next_z = critic_forward(next_states, next_actions,
                                         target_parameters)
                sorted_z = torch.sort(next_z.flatten(1), dim=1).values[:, :keep]
                target = rewards + masks * sorted_z
                return quantile_huber_loss(current_z, target.detach())
        else:
            def critic_loss(states, actions, rewards, next_states,
                            next_actions, masks, parameters, target_parameters):
                current_q1, current_q2 = critic_forward(
                    states, actions, parameters
                )
                target_q1, target_q2 = critic_forward(
                    next_states, next_actions, target_parameters
                )
                target_q = rewards + masks * torch.minimum(target_q1, target_q2)
                target_q = target_q.detach()
                return (F.mse_loss(current_q1, target_q) +
                        F.mse_loss(current_q2, target_q))

        self._critic_value_graph = critic_value
        self._critic_loss_graph = critic_loss
        self._actor_loss_graph = self.actor.loss

        # The complete DDPM step is reused by actor and target. Compiling the
        # posterior/noise update together with the denoiser eliminates the
        # small eager kernels that otherwise run 20 times per sample.
        # A fully-unrolled 20-step graph was measured and was substantially
        # slower due to code size and register/memory pressure.
        self._sample_step_graph = self.actor.functional_sample_step

        train_sample = self.train_sample
        chosen = self.chosen
        action_dim = self.actor.action_dim

        def select_augmented_actions(states, candidate_actions, parameters):
            """Rank all generated candidates and gather the training batch."""
            raw_batch_size = states.shape[0]
            expanded_states = states.repeat(train_sample, 1)
            q = critic_forward(
                expanded_states, candidate_actions, parameters
            )
            if self.critic_type == 'quantile':
                q = q.mean(dim=(1, 2)).unsqueeze(-1)
            else:
                q = torch.minimum(q[0], q[1])
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
            return (selected_states, selected_actions,
                    selected_q.reshape(raw_batch_size * chosen, 1),
                    v, mean, std)

        self._select_augmented_actions_graph = select_augmented_actions

        if self.compile_enabled:
            compile_options = dict(fullgraph=True, dynamic=True,
                                   backend=self.compile_backend)
            if self.compile_backend == 'inductor':
                compile_options['mode'] = self.compile_mode
            self._critic_value_graph = torch.compile(
                self._critic_value_graph, **compile_options
            )
            self._critic_loss_graph = torch.compile(
                self._critic_loss_graph, **compile_options
            )
            self._actor_loss_graph = torch.compile(
                self._actor_loss_graph, **compile_options
            )
            self._sample_step_graph = torch.compile(
                self._sample_step_graph, **compile_options
            )
            self._select_augmented_actions_graph = torch.compile(
                self._select_augmented_actions_graph, **compile_options
            )
        self.actor.set_compiled_sample_step(self._sample_step_graph)
        self.actor_target.set_compiled_sample_step(self._sample_step_graph)

    def _critic_value(self, critic, state, action):
        return self._critic_value_graph(
            state, action, tuple(critic.parameters())
        )

    def _online_q(self, state, action):
        return self._critic_value(self.critic, state, action)

    def _target_q(self, state, action):
        return self._critic_value(self.critic_target, state, action)

    def append_memory(self, state, action, reward, next_state, mask):
        action = (action - self.action_bias) / self.action_scale

        self.memory.append(state, action, reward, next_state, mask)

    def sample_action(self, state, eval=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

        normal = False
        if not eval and torch.rand(1).item() <= self.epsilon:
            normal = True

        action = self.actor(state, eval, q_func=self._online_q, normal=normal).cpu().data.numpy().flatten()
        action = action.clip(-1, 1)
        action = action * self.action_scale + self.action_bias
        return action

    def action_aug(self, batch_size, log_writer, return_mean_std=False):
        states, actions, rewards, next_states, masks = self.memory.sample(batch_size)
        expanded_states = states.repeat(self.train_sample, 1)
        candidate_actions = self.actor.p_sample_loop(
            expanded_states,
            (expanded_states.shape[0], self.action_dim),
        ).clamp_(-1.0, 1.0)
        states, best_actions, q, v, mean, std = (
            self._select_augmented_actions_graph(
                states, candidate_actions, tuple(self.critic.parameters())
            )
        )

        if return_mean_std:
            return states, best_actions, (q, v), (mean, std)
        else:
            return states, best_actions, (q, v)

    def train(self, t, iterations, batch_size=256, log_writer=None):
        for itr in range(iterations):
            # Sample replay buffer / batch
            states, actions, rewards, next_states, masks = self.memory.sample(batch_size)

            """ Q Training """
            with torch.no_grad():
                next_actions = self.actor_target(
                    next_states, eval=False, q_func=self._target_q
                )

            critic_loss = self._critic_loss_graph(
                states, actions, rewards, next_states, next_actions, masks,
                tuple(self.critic.parameters()),
                tuple(self.critic_target.parameters())
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
                states, best_actions, qv, (mean, std) = self.action_aug(
                    batch_size, log_writer, return_mean_std=True
                )

                if self.policy_type == 'Diffusion' and self.weighted:
                    q, v = qv
                    # print("q shape", q.shape)
                    self.running_q_std += self.alpha_std * (std - self.running_q_std)
                    self.running_q_mean += self.alpha_mean * (mean - self.running_q_mean)
                    # q.clamp_(-self.q_neg).add_(self.q_neg)
                    q = eval(self.q_transform)(q, q_neg=self.q_neg, cut=self.cut, running_q_std=self.running_q_std, beta=self.beta,
                                               running_q_mean=self.running_q_mean, v=v, batch_size=batch_size, chosen=self.chosen)
                    if self.entropy_alpha > 0.0:
                        rand_states = states.unsqueeze(0).expand(10, -1, -1).contiguous().view(batch_size*self.chosen*10, -1)
                        rand_policy_actions = torch.empty(batch_size * self.chosen * 10, actions.shape[-1], device=self.device).uniform_(
                            -1, 1)
                        rand_q = q.unsqueeze(0).expand(10, -1, -1).contiguous().view(batch_size*self.chosen*10, -1) * self.entropy_alpha

                        best_actions = torch.cat([best_actions, rand_policy_actions], dim=0)
                        states = torch.cat([states, rand_states], dim=0)
                        q = torch.cat([q, rand_q], dim=0)
                    # q[q<1.0] = 1.0
                    # q = torch.clip(q / self.running_avg_qnorm, -6 ,6)
                    # expq = torch.exp(self.beta * q)
                    # expq[expq<=expq.quantile(0.95)] = 0.0
                    # if itr % 10000 == 0 : print(expq, itr)
                    # print("expq", expq.shape)
                    actor_loss = self._actor_loss_graph(best_actions, states, weights=q)
                else:
                    actor_loss = self._actor_loss_graph(best_actions, states)

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
