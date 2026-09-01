"""Measure the real QVPO hot path and project the first 100k env steps."""
import argparse
import time
from argparse import Namespace

import numpy as np
import torch

from agent.qvpo import QVPO
from agent.replay_memory import ReplayMemory


class ActionSpace:
    shape = (3,)
    high = np.ones(3, dtype=np.float32)
    low = -high


def make_args(sampling_precision):
    return Namespace(
        policy_type='Diffusion', noise_ratio=1.0, beta_schedule='cosine',
        n_timesteps=20, behavior_sample=4, target_sample=4, eval_sample=32,
        deterministic=False, beta=1.0, alpha_mean=0.001, alpha_std=0.001,
        chosen=1, q_neg=0.0, weighted=True, train_sample=64,
        q_transform='qadv', policy_freq=1, cut=1.0,
        epsilon=0.0, entropy_alpha=0.02, diffusion_lr=1e-4,
        ac_grad_norm=2.0, tau=0.005,
        update_actor_target_every=1, critic_lr=3e-4,
        critic_type='quantile', n_quantiles=25, n_nets=5,
        top_quantiles_to_drop_per_net=2, compile=True,
        compile_mode='reduce-overhead', compile_backend='inductor',
        sampling_precision=sampling_precision,
    )


def filled_memory(capacity, device):
    rng = np.random.default_rng(0)
    replay = ReplayMemory(11, 3, capacity, device)
    replay.states[:] = rng.standard_normal(replay.states.shape)
    replay.actions[:] = rng.uniform(-1, 1, replay.actions.shape)
    replay.rewards[:] = rng.standard_normal(replay.rewards.shape)
    replay.next_states[:] = rng.standard_normal(replay.next_states.shape)
    replay.masks[:] = 0.99
    replay.idx, replay.full = 0, True

    return replay


def synchronized_seconds(fn, iterations):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iterations


def hours(seconds):
    return seconds / 3600


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--update-samples', type=int, default=20)
    parser.add_argument('--action-samples', type=int, default=50)
    parser.add_argument('--eval-horizon', type=int, default=1000,
                        help='assumed steps per evaluation episode')
    parser.add_argument('--sampling-precision', default='float16',
                        choices=('float16', 'float32'))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required')
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda')
    replay = filled_memory(20_000, device)
    agent = QVPO(
        make_args(args.sampling_precision), 11, ActionSpace(), replay, device
    )
    state = np.zeros(11, dtype=np.float32)

    warmup_start = time.perf_counter()
    agent.train(10_000, 1, batch_size=args.batch_size)
    agent.sample_action(state, eval=False)
    agent.sample_action(state, eval=True)
    # reduce-overhead records CUDA Graphs on calls after compilation. Finish
    # those recordings before measuring steady-state updates/actions.
    for _ in range(10):
        agent.train(10_001, 1, batch_size=args.batch_size)
    for _ in range(3):
        agent.sample_action(state, eval=False)
        agent.sample_action(state, eval=True)
    torch.cuda.synchronize()
    compile_warmup = time.perf_counter() - warmup_start

    update_s = synchronized_seconds(
        lambda: agent.train(10_001, 1, batch_size=args.batch_size),
        args.update_samples,
    )
    behavior_s = synchronized_seconds(
        lambda: agent.sample_action(state, eval=False), args.action_samples
    )
    eval_s = synchronized_seconds(
        lambda: agent.sample_action(state, eval=True), args.action_samples
    )

    # main.py: 10k random actions, then 90,001 updates and about 90k policy
    # actions. Evaluation runs every 10k steps: 10 runs x 10 episodes.
    updates = 100_000 - 10_000 + 1
    behavior_actions = 100_000 - 10_000
    eval_actions = 10 * 10 * args.eval_horizon
    training_seconds = updates * update_s + behavior_actions * behavior_s
    evaluation_seconds = eval_actions * eval_s
    total_seconds = compile_warmup + training_seconds + evaluation_seconds

    print(f'compile warm-up:       {compile_warmup:.2f} s')
    print(f'one training update:   {update_s * 1000:.3f} ms')
    print(f'one behavior action:   {behavior_s * 1000:.3f} ms')
    print(f'one evaluation action: {eval_s * 1000:.3f} ms')
    print(f'100k train/collect:    {hours(training_seconds):.2f} h')
    print(f'evaluation allowance:  {hours(evaluation_seconds):.2f} h '
          f'({args.eval_horizon} steps/episode)')
    print(f'projected total:       {hours(total_seconds):.2f} h')
    print('Note: MuJoCo CPU stepping, logging, and checkpoint I/O are not included.')


if __name__ == '__main__':
    main()
