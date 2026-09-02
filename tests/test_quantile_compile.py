from argparse import Namespace

import numpy as np
import torch

from agent.model import CriticQR, Model
from agent.qvpo import QVPO


class ActionSpace:
    shape = (3,)
    high = np.ones(3, dtype=np.float32)
    low = -high


class Memory:
    def __init__(self, batch, state_dim=11, action_dim=3):
        self.batch = batch
        self.values = (
            torch.randn(batch, state_dim),
            torch.randn(batch, action_dim).clamp_(-1, 1),
            torch.randn(batch, 1),
            torch.randn(batch, state_dim),
            torch.rand(batch, 1),
        )

    def sample(self, batch_size):
        assert batch_size == self.batch
        return self.values


def make_args(**overrides):
    values = dict(
        policy_type='Diffusion', noise_ratio=1.0, beta_schedule='cosine',
        n_timesteps=4, behavior_sample=2, target_sample=2, eval_sample=4,
        deterministic=False, beta=1.0, alpha_mean=0.001, alpha_std=0.001,
        chosen=1, q_neg=0.0, weighted=True, train_sample=2,
        q_transform='qadv', policy_freq=1, cut=1.0,
        entropy_alpha=0.0, diffusion_lr=1e-4,
        ac_grad_norm=2.0, tau=0.005,
        update_actor_target_every=1, critic_lr=3e-4,
        critic_type='quantile', n_quantiles=7, n_nets=3,
        top_quantiles_to_drop_per_net=2, compile=False,
        compile_mode='default', compile_backend='inductor',
        sampling_precision='float16',
    )
    values.update(overrides)
    return Namespace(**values)


def test_stateless_quantile_forward_matches_module():
    torch.manual_seed(0)
    critic = CriticQR(11, 3, hidden_dim=32, n_quantiles=7, n_nets=3)
    state, action = torch.randn(8, 11), torch.randn(8, 3)
    expected = critic(state, action)
    actual = critic.functional_forward(state, action, tuple(critic.parameters()))
    torch.testing.assert_close(actual, expected)


def test_packed_quantile_critic_migrates_legacy_checkpoint():
    torch.manual_seed(0)
    critic = CriticQR(11, 3, hidden_dim=32, n_quantiles=7, n_nets=3)
    state, action = torch.randn(8, 11), torch.randn(8, 3)
    expected = critic(state, action)
    legacy = {}
    for layer, module_index in enumerate((0, 2, 4, 6)):
        for net in range(critic.n_nets):
            legacy[f'nets.{net}.{module_index}.weight'] = (
                critic.layers[2 * layer][net].detach().clone()
            )
            legacy[f'nets.{net}.{module_index}.bias'] = (
                critic.layers[2 * layer + 1][net].detach().clone()
            )
    restored = CriticQR(11, 3, hidden_dim=32, n_quantiles=7, n_nets=3)
    restored.load_state_dict(legacy)
    torch.testing.assert_close(restored(state, action), expected)
    assert len(tuple(restored.parameters())) == 8


def test_stateless_denoiser_forward_matches_module():
    torch.manual_seed(0)
    model = Model(11, 3, hidden_size=32, time_dim=8)
    x, state = torch.randn(8, 3), torch.randn(8, 11)
    time = torch.randint(0, 4, (8,))
    actual = model.functional_forward(x, time, state, tuple(model.parameters()))
    torch.testing.assert_close(actual, model(x, time, state))


def test_quantile_training_loss_has_finite_gradients():
    agent = QVPO(make_args(), 11, ActionSpace(), None,
                 torch.device('cpu'))
    batch = 8
    states, next_states = torch.randn(batch, 11), torch.randn(batch, 11)
    actions = torch.randn(batch, 3)
    rewards, masks = torch.randn(batch, 1), torch.rand(batch, 1)
    loss = agent._critic_train_loss_graph(
        states, actions, rewards, next_states, masks,
        tuple(agent.critic.parameters()),
        tuple(agent.critic_target.parameters()),
        tuple(agent.actor_target.model.parameters()),
        agent.actor_target.noise_ratio,
    )
    loss.backward()
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in agent.critic.parameters())


def test_direct_candidate_quantile_target_matches_action_recompute():
    torch.manual_seed(0)
    agent = QVPO(make_args(), 11, ActionSpace(), None,
                 torch.device('cpu'))
    batch = 5
    samples = agent.actor_target.behavior_sample
    states, next_states = torch.randn(batch, 11), torch.randn(batch, 11)
    actions = torch.randn(batch, 3)
    rewards, masks = torch.randn(batch, 1), torch.rand(batch, 1)

    torch.manual_seed(123)
    actual = agent._critic_train_loss_graph(
        states, actions, rewards, next_states, masks,
        tuple(agent.critic.parameters()),
        tuple(agent.critic_target.parameters()),
        tuple(agent.actor_target.model.parameters()),
        agent.actor_target.noise_ratio,
    )

    torch.manual_seed(123)
    expanded_next_states = next_states.repeat(samples, 1)
    initial_noise = torch.randn(samples * batch, 3)
    candidates = agent.actor_target.functional_sample_loop(
        expanded_next_states, initial_noise,
        tuple(agent.actor_target.model.parameters()),
        agent.actor_target.noise_ratio,
    ).clamp(-1, 1)
    candidate_z = agent.critic_target(expanded_next_states, candidates)
    candidate_q = candidate_z.mean(dim=(1, 2)).view(
        samples, batch
    ).transpose(0, 1)
    best = candidate_q.argmax(dim=1)
    candidate_actions = candidates.view(
        samples, batch, 3
    ).permute(1, 0, 2)
    selected_actions = candidate_actions.gather(
        1, best[:, None, None].expand(-1, 1, 3)
    ).squeeze(1)
    selected_z = agent.critic_target(next_states, selected_actions)
    keep = agent.quantiles_total - (
        agent.top_quantiles_to_drop_per_net * agent.n_nets
    )
    target = rewards + masks * torch.sort(
        selected_z.flatten(1), dim=1
    ).values[:, :keep]
    expected = __import__(
        'agent.helpers', fromlist=['quantile_huber_loss']
    ).quantile_huber_loss(
        agent.critic(states, actions), target.detach()
    )
    torch.testing.assert_close(actual, expected)


def test_actor_train_loss_does_not_backpropagate_into_critic():
    agent = QVPO(make_args(), 11, ActionSpace(), None,
                 torch.device('cpu'))
    batch = 4
    states = torch.randn(batch, 11)
    loss, running_mean, running_std = agent._actor_train_loss_graph(
        states, tuple(agent.critic.parameters()),
        tuple(agent.actor.model.parameters()), agent.actor.noise_ratio,
        agent.running_q_mean, agent.running_q_std,
        agent._entropy_alpha_tensor,
    )
    loss.backward()
    assert all(p.grad is not None for p in agent.actor.parameters())
    assert all(p.grad is None for p in agent.critic.parameters())
    assert not running_mean.requires_grad
    assert not running_std.requires_grad


def test_one_quantile_training_iteration_runs():
    memory = Memory(batch=4)
    agent = QVPO(make_args(), 11, ActionSpace(), memory,
                 torch.device('cpu'))
    agent.train(t=1, iterations=1, batch_size=4)
    assert agent.step == 1


def test_entropy_schedule_updates_graph_tensor_without_reallocation():
    agent = QVPO(make_args(entropy_alpha=0.02), 11, ActionSpace(), None,
                 torch.device('cpu'))
    storage = agent._entropy_alpha_tensor.data_ptr()
    agent.entropy_alpha = 0.0125
    assert agent.entropy_alpha == 0.0125
    assert agent._entropy_alpha_tensor.data_ptr() == storage
    torch.testing.assert_close(
        agent._entropy_alpha_tensor, torch.tensor(0.0125)
    )


def test_fullgraph_training_iteration_runs():
    memory = Memory(batch=4)
    agent = QVPO(
        make_args(compile=True, compile_backend='eager'),
        11, ActionSpace(), memory, torch.device('cpu')
    )
    agent.train(t=1, iterations=1, batch_size=4)
    assert agent.step == 1


def test_fullgraph_interaction_action_runs_without_epsilon():
    agent = QVPO(
        make_args(compile=True, compile_backend='eager'),
        11, ActionSpace(), None, torch.device('cpu')
    )
    action = agent.sample_action(np.zeros(11, dtype=np.float32))
    assert action.shape == (3,)
    assert np.isfinite(action).all()
    assert not hasattr(agent, 'epsilon')


def test_functional_sampler_matches_stepwise_sampler():
    agent = QVPO(make_args(), 11, ActionSpace(), None,
                 torch.device('cpu'))
    state = torch.randn(6, 11)
    initial_noise = torch.randn(6, 3)
    parameters = tuple(agent.actor.model.parameters())

    torch.manual_seed(123)
    expected = initial_noise.clone()
    for i in reversed(range(agent.actor.n_timesteps)):
        timesteps = torch.full((6,), i, dtype=torch.long)
        expected = agent.actor.p_sample(expected, timesteps, state)

    torch.manual_seed(123)
    actual = agent.actor.functional_sample_loop(
        state, initial_noise.clone(), parameters, agent.actor.noise_ratio
    )
    torch.testing.assert_close(actual, expected)
