import argparse
import csv

import numpy as np
import torch

from agent.qvpo import QVPO
from agent.replay_memory import ReplayMemory

try:
    from tensorboardX import SummaryWriter
except ImportError:
    class SummaryWriter:
        """No-op fallback when TensorBoard logging is not installed."""
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass
try:
    import gymnasium as gym
except ImportError:
    import gym
import os
from logger import Logger
import datetime

def readParser():
    parser = argparse.ArgumentParser(description='Diffusion Policy')
    parser.add_argument('--env_name', default="Hopper-v3",
                        help='Mujoco Gym environment (default: Hopper-v3)')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed (default: 0)')

    parser.add_argument('--num_steps', type=int, default=1000000, metavar='N',
                        help='env timesteps (default: 1000000)')
    parser.add_argument('--eval_interval', type=int, default=10000)
    parser.add_argument('--eval_episodes', type=int, default=10)
    parser.add_argument('--metrics_path', type=str, default=None,
                        help='append evaluation metrics to this CSV file')

    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--update_actor_target_every', type=int, default=1, metavar='N',
                        help='update actor target per iteration (default: 1)')

    parser.add_argument("--policy_type", type=str, default="Diffusion", metavar='S',
                        help="Diffusion, VAE or MLP")
    parser.add_argument("--beta_schedule", type=str, default="cosine", metavar='S',
                        help="linear, cosine or vp")
    parser.add_argument('--n_timesteps', type=int, default=20, metavar='N',
                        help='diffusion timesteps (default: 20)')
    parser.add_argument('--diffusion_lr', type=float, default=0.0001, metavar='G',
                        help='diffusion learning rate (default: 0.0001)')
    parser.add_argument('--critic_lr', type=float, default=0.0003, metavar='G',
                        help='critic learning rate (default: 0.0003)')
    parser.add_argument('--critic_type', choices=('quantile', 'double'),
                        default='quantile',
                        help='distributional quantile ensemble or twin-Q critic')
    parser.add_argument('--n_quantiles', type=int, default=25)
    parser.add_argument('--n_nets', type=int, default=5)
    parser.add_argument('--top_quantiles_to_drop_per_net', type=int, default=2,
                        help='TQC truncation applied to every target ensemble')
    parser.add_argument('--compile', dest='compile', action='store_true',
                        default=True,
                        help='enable four runtime torch.compile full graphs (default)')
    parser.add_argument('--no_compile', dest='compile', action='store_false',
                        help='disable torch.compile for debugging/benchmarking')
    parser.add_argument('--compile_mode', default='reduce-overhead',
                        choices=('default', 'reduce-overhead', 'max-autotune'),
                        help='torch.compile mode (default: reduce-overhead)')
    parser.add_argument('--compile_backend', default='inductor',
                        help='torch.compile backend (default: inductor)')
    parser.add_argument('--sampling_precision', default='float16',
                        choices=('float16', 'float32'),
                        help='no-grad DDPM sampling precision (default: float16)')
    parser.add_argument('--noise_ratio', type=float, default=1.0, metavar='G',
                        help='noise ratio in sample process (default: 1.0)')
    parser.add_argument('--ac_grad_norm', type=float, default=2.0, metavar='G',
                        help='actor and critic grad norm (default: 1.0)')

    parser.add_argument('--cuda', default='cuda:0',
                        help='run on CUDA (default: cuda:0)')

    parser.add_argument('--alpha_mean', type=float, default=0.001, metavar='G',
                        help='running mean update weight (default: 0.1)')

    parser.add_argument('--alpha_std', type=float, default=0.001, metavar='G',
                        help='running std update weight (default: 0.001)')

    parser.add_argument('--beta', type=float, default=1.0, metavar='G',
                        help='expQ weight (default: 1.0)')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        default=True, help='enable weighted training (default)')
    parser.add_argument('--no_weighted', dest='weighted', action='store_false',
                        help='disable weighted training for ablation')

    parser.add_argument('--train_sample', type=int, default=64, metavar='N',
                        help='train_sample (default: 64)')

    parser.add_argument('--chosen', type=int, default=1, metavar='N', help="chosen actions (default:1)")

    parser.add_argument('--q_neg', type=float, default=0.0, metavar='G', help="q_neg (default: 0.0)")

    parser.add_argument('--behavior_sample', type=int, default=4, metavar='N', help="behavior_sample (default: 1)")
    parser.add_argument('--target_sample', type=int, default=4, metavar='N', help="target_sample (default: behavior sample)")

    parser.add_argument('--eval_sample', type=int, default=32, metavar='N', help="eval_sample (default: 512)")

    parser.add_argument('--deterministic', action="store_true", help="deterministic mode")

    parser.add_argument('--q_transform', type=str, default='qadv', metavar='S', help="q_transform (default: qrelu)")

    parser.add_argument('--policy_freq', type=int, default=1, metavar='N', help="policy_freq (default: 1)")

    parser.add_argument('--cut', type=float, default=1.0, metavar='G', help="cut (default: 1.0)")
    parser.add_argument('--times', type=int, default=1, metavar='N', help="times (default: 1)")

    parser.add_argument('--epsilon', type=float, default=0.0, metavar='G', help="eps greedy (default: 0.0)")
    parser.add_argument('--entropy_alpha', type=float, default=0.02, metavar='G', help="entropy_alpha (default: 0.02)")


    return parser.parse_args()


def reset_env(env, seed=None):
    result = env.reset(seed=seed) if seed is not None else env.reset()
    return result[0] if isinstance(result, tuple) else result


def step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        next_state, reward, terminated, truncated, info = result
        return next_state, reward, terminated or truncated, terminated, info
    next_state, reward, done, info = result
    return next_state, reward, done, done, info


def evaluate(env, agent, steps, episodes=10):
    returns = np.zeros((episodes,), dtype=np.float32)

    for i in range(episodes):
        state = reset_env(env)
        episode_reward = 0.
        done = False
        while not done:
            action = agent.sample_action(state, eval=True)
            next_state, reward, done, _, _ = step_env(env, action)
            episode_reward += reward
            state = next_state
        returns[i] = episode_reward

    mean_return = np.mean(returns)
    std_return = np.std(returns)

    print('-' * 60)
    print(f'Num steps: {steps:<5}  '
          f'reward: {mean_return:<5.1f}  '
          f'std: {std_return:<5.1f}')
    print(returns)
    print('-' * 60)
    return mean_return


def main(args=None, logger=None, id=None):

    device = torch.device(args.cuda)
    if args.compile and device.type == 'cuda':
        # Let Inductor use TensorFloat32 tensor cores on Ampere and newer GPUs.
        torch.set_float32_matmul_precision('high')

    dir = "record"
    # dir = "test"
    log_dir = os.path.join(dir, f'{args.env_name}', f'policy_type={args.policy_type}',
                           f'seed={args.seed}')
    writer = SummaryWriter(log_dir)

    # Initial environment
    env = gym.make(args.env_name)
    eval_env = gym.make(args.env_name)
    state_size = int(np.prod(env.observation_space.shape))
    action_size = int(np.prod(env.action_space.shape))
    print(action_size)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    reset_env(env, args.seed)
    reset_env(eval_env, args.seed + 10000)
    env.action_space.seed(args.seed)
    eval_env.action_space.seed(args.seed + 10000)

    memory_size = 1e6
    num_steps = args.num_steps
    start_steps = 10000
    eval_interval = args.eval_interval
    updates_per_step = 1
    batch_size = args.batch_size
    log_interval = 10

    memory = ReplayMemory(state_size, action_size, memory_size, device)
    agent = QVPO(args, state_size, env.action_space, memory, device)

    steps = 0
    episodes = 0
    best_result = -float('inf')

    while steps < num_steps:
        episode_reward = 0.
        episode_steps = 0
        done = False
        state = reset_env(env)
        episodes += 1
        while not done:
            if start_steps > steps:
                action = env.action_space.sample()
            else:
                action = agent.sample_action(state, eval=False)
            next_state, reward, done, terminated, _ = step_env(env, action)

            mask = 0.0 if terminated else args.gamma

            steps += 1
            episode_steps += 1
            episode_reward += reward

            agent.append_memory(state, action, reward, next_state, mask)

            if steps >= start_steps:
                agent.train(steps, updates_per_step, batch_size=batch_size, log_writer=writer)
                agent.entropy_alpha = min(args.entropy_alpha, max(0.002, args.entropy_alpha-steps/num_steps*args.entropy_alpha))

            if steps % eval_interval == 0:
                tmp_result = evaluate(
                    eval_env, agent, steps, episodes=args.eval_episodes
                )
                writer.add_scalar('reward/eval', tmp_result, steps)
                if args.metrics_path is not None:
                    metrics_dir = os.path.dirname(args.metrics_path)
                    if metrics_dir:
                        os.makedirs(metrics_dir, exist_ok=True)
                    new_file = not os.path.exists(args.metrics_path)
                    with open(args.metrics_path, 'a', newline='') as metrics:
                        csv_writer = csv.writer(metrics)
                        if new_file:
                            csv_writer.writerow(
                                ['environment', 'seed', 'step', 'mean_reward']
                            )
                        csv_writer.writerow(
                            [args.env_name, args.seed, steps, tmp_result]
                        )
                if tmp_result > best_result:
                    best_result = tmp_result
                    agent.save_model(os.path.join('./results', prefix + '_' + name), id=id)
                # self.save_models()

            state = next_state

        # if episodes % log_interval == 0:
        #     writer.add_scalar('reward/train', episode_reward, steps)

        print(f'episode: {episodes:<4}  '
              f'episode steps: {episode_steps:<4}  '
              f'reward: {episode_reward:<5.1f}')

        if logger is not None:
            for i in range(episode_steps):
                logger.add(epoch=steps-episode_steps+i, reward=episode_reward)


if __name__ == "__main__":
    args = readParser()
    if args.target_sample == -1:
        args.target_sample = args.behavior_sample


    ## settings
    prefix = 'qvpo'
    name = args.env_name
    keys = ("epoch", "reward")
    times = args.times
    id = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")
    logger = Logger(name=name, keys=keys, max_epochs=int(args.num_steps)+2100, times=times, config=args, path=os.path.join('./results', prefix + '_' + name), id=id)


    ## run
    for time in range(times):
        main(args, logger=logger, id=id+"_"+str(time))

    logger.save(os.path.join('./results', prefix + '_' + name), id=id)
