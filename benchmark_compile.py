"""Verify full-graph count and benchmark the five QVPO compute graphs.

Run after installing the project requirements:
    python benchmark_compile.py --batch-size 256 --iterations 100
"""
import argparse
import copy
import time
from argparse import Namespace

import numpy as np
import torch

from agent.qvpo import QVPO


class ActionSpace:
    shape = (3,)
    high = np.ones(3, dtype=np.float32)
    low = -high


def agent_args(compiled, mode, backend):
    return Namespace(
        policy_type='Diffusion', noise_ratio=1.0, beta_schedule='cosine',
        n_timesteps=20, behavior_sample=4, target_sample=4, eval_sample=32,
        deterministic=False, beta=1.0, alpha_mean=0.001, alpha_std=0.001,
        chosen=1, q_neg=0.0, weighted=True, train_sample=64,
        q_transform='qadv', policy_freq=1, cut=1.0,
        epsilon=0.0, entropy_alpha=0.0, diffusion_lr=1e-4,
        ac_grad_norm=2.0, tau=0.005,
        update_actor_target_every=1, critic_lr=3e-4,
        critic_type='quantile', n_quantiles=25, n_nets=5,
        top_quantiles_to_drop_per_net=2, compile=compiled,
        compile_mode=mode, compile_backend=backend,
        sampling_precision='float16',
    )


def inputs(batch, device):
    return dict(
        states=torch.randn(batch, 11, device=device),
        actions=torch.randn(batch, 3, device=device).clamp_(-1, 1),
        rewards=torch.randn(batch, 1, device=device),
        next_states=torch.randn(batch, 11, device=device),
        next_actions=torch.randn(batch, 3, device=device).clamp_(-1, 1),
        masks=torch.rand(batch, 1, device=device),
        times=torch.randint(0, 20, (batch,), device=device),
    )


def exercise(agent, data, backward=True):
    params = tuple(agent.critic.parameters())
    target_params = tuple(agent.critic_target.parameters())
    with torch.no_grad():
        value = agent._critic_value_graph(
            data['states'], data['actions'], params
        ).clone()
        ranked_value = agent._critic_value_graph(
            data['states'], data['actions'], target_params
        ).clone()
    critic_loss = agent._critic_loss_graph(
        data['states'], data['actions'], data['rewards'], data['next_states'],
        data['next_actions'], data['masks'], params, target_params,
    )
    sampled = agent._sample_step_graph(
        data['actions'], data['times'], data['states'],
        tuple(agent.actor.model.parameters()),
        agent.actor.noise_ratio,
    )
    selected = agent._select_augmented_actions_graph(
        data['states'],
        data['actions'].repeat(agent.train_sample, 1),
        params,
    )
    actor_loss = agent._actor_loss_graph(data['actions'], data['states'])
    if backward:
        agent.critic_optimizer.zero_grad(set_to_none=True)
        agent.actor_optimizer.zero_grad(set_to_none=True)
        (critic_loss + actor_loss).backward()
    return value + ranked_value, critic_loss, sampled, actor_loss, selected


def measure(agent, data, iterations):
    for _ in range(3):
        exercise(agent, data)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        exercise(agent, data)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / iterations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--mode', default='reduce-overhead',
                        choices=('default', 'reduce-overhead', 'max-autotune'))
    parser.add_argument('--backend', default='inductor')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required for a meaningful speed benchmark')

    device = torch.device('cuda')
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(0)
    eager = QVPO(agent_args(False, args.mode, args.backend), 11, ActionSpace(), None,
                 device)
    eager_state = copy.deepcopy(eager.actor.state_dict())
    eager_critic_state = copy.deepcopy(eager.critic.state_dict())

    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    compiled = QVPO(agent_args(True, args.mode, args.backend), 11, ActionSpace(), None,
                    device)
    compiled.actor.load_state_dict(eager_state)
    compiled.critic.load_state_dict(eager_critic_state)
    compiled.critic_target.load_state_dict(eager_critic_state)
    data = inputs(args.batch_size, device)

    # First pass proves fullgraph capture; any graph break raises immediately.
    torch.manual_seed(123)
    compiled_outputs = exercise(compiled, data, backward=False)
    torch.manual_seed(123)
    eager_outputs = exercise(eager, data, backward=False)
    # Sampling and actor loss intentionally use a different Philox stream under
    # Inductor. Critic values/losses are deterministic.
    for index in (0, 1):
        actual, expected = compiled_outputs[index], eager_outputs[index]
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    for actual, expected in zip(compiled_outputs[4], eager_outputs[4]):
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    for index in (2, 3):
        assert torch.isfinite(compiled_outputs[index]).all()
        assert torch.isfinite(eager_outputs[index]).all()
    unique_graphs = torch._dynamo.utils.counters['stats']['unique_graphs']

    eager_ms = measure(eager, data, args.iterations)
    compiled_ms = measure(compiled, data, args.iterations)
    print(f'fullgraph runtime graphs: {unique_graphs} (design minimum: 5)')
    print(f'eager:    {eager_ms:.3f} ms/iteration')
    print(f'compiled: {compiled_ms:.3f} ms/iteration')
    print(f'speedup:  {eager_ms / compiled_ms:.2f}x')


if __name__ == '__main__':
    main()
