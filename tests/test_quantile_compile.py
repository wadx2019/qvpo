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
        epsilon=0.0, entropy_alpha=0.0, diffusion_lr=1e-4,
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
    actions, next_actions = torch.randn(batch, 3), torch.randn(batch, 3)
    rewards, masks = torch.randn(batch, 1), torch.rand(batch, 1)
    loss = agent._critic_loss_graph(
        states, actions, rewards, next_states, next_actions, masks,
        tuple(agent.critic.parameters()),
        tuple(agent.critic_target.parameters()),
    )
    loss.backward()
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in agent.critic.parameters())


def test_one_quantile_training_iteration_runs():
    memory = Memory(batch=4)
    agent = QVPO(make_args(), 11, ActionSpace(), memory,
                 torch.device('cpu'))
    agent.train(t=1, iterations=1, batch_size=4)
    assert agent.step == 1


def test_fullgraph_training_iteration_runs():
    memory = Memory(batch=4)
    agent = QVPO(
        make_args(compile=True, compile_backend='eager'),
        11, ActionSpace(), memory, torch.device('cpu')
    )
    agent.train(t=1, iterations=1, batch_size=4)
    assert agent.step == 1


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
