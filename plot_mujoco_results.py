"""Plot evaluation CSV files emitted by main.py for the MuJoCo suite."""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('metrics_dir', type=Path)
    parser.add_argument('--output', type=Path, default=Path('mujoco_1m_rewards.png'))
    args = parser.parse_args()

    series = {}
    for path in sorted(args.metrics_dir.glob('*.csv')):
        with path.open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        if rows:
            label = rows[0]['environment'].removesuffix('-v4')
            series[label] = (
                [int(row['step']) for row in rows],
                [float(row['mean_reward']) for row in rows],
            )
    if not series:
        raise SystemExit(f'No metric CSV files found in {args.metrics_dir}')

    plt.figure(figsize=(10, 6), dpi=160)
    max_step = max(max(steps) for steps, _ in series.values())
    for label, (steps, rewards) in series.items():
        plt.plot(steps, rewards, linewidth=1.8, label=label)
    plt.xlabel('Environment steps')
    plt.ylabel('Mean evaluation reward (10 episodes)')
    title = (
        'QVPO MuJoCo 1M-step learning curves'
        if max_step >= 1_000_000
        else f'QVPO MuJoCo training progress (through {max_step // 1000}k steps)'
    )
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output)
    print(args.output.resolve())


if __name__ == '__main__':
    main()
