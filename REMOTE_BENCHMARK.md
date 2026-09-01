# Remote CUDA benchmark

Run on 2026-09-01 with the default distributional quantile critic and
`torch.compile(mode="reduce-overhead", fullgraph=True, dynamic=True)`.

- GPU: NVIDIA GeForce RTX 4090 (24 GB)
- PyTorch: 2.4.1+cu121
- Triton: 3.0.0
- Batch size: 256
- Fullgraph runtime graphs: 5
- Tests: 6 passed
- Default action augmentation: 64 candidates, always enabled
- Default weighted actor loss: enabled
- DDPM sampling precision: FP16; critic/training math: FP32

Steady-state results (compile warm-up excluded):

| Iterations | Eager | Compiled | Speedup |
|---:|---:|---:|---:|
| 100 | 20.361 ms/iteration | 5.231 ms/iteration | 3.89x |

This workload includes critic inference/loss, a complete fused DDPM step,
actor loss, and quantile ranking/gather for all 64 augmentation candidates.
The removed action-gradient loop is not part of the measurement. A single
fully-unrolled 20-step sampler was rejected after measurement because it was
slower and took six minutes to compile; the optimized implementation replays
one complete compiled DDPM-step graph.

Command:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark_compile.py \
  --batch-size 256 --iterations 100
```

## Projected first 100k environment steps

The full `agent.train()` path was measured with the new main-program defaults:
batch size 256, 20 diffusion steps, 64-candidate augmentation, weighted actor
loss, behavior sample 4, evaluation sample 32, quantile critic, FP16 no-grad
sampling, and `reduce-overhead` compilation.

Two steady-state runs after CUDA Graph recording measured:

- compile and CUDA Graph warm-up: 39.09 / 39.90 seconds
- one complete training update: 27.437 / 28.946 ms
- one behavior action: 7.432 / 7.756 ms
- one evaluation action: 7.409 / 7.713 ms
- 90,001 updates plus 90,000 policy actions: 0.87 / 0.92 hours
- ten evaluations, assuming 10 episodes x 1,000 steps: 0.21 hours
- projected total: **1.09–1.14 hours**

The FP32 sampling control measured 29.856 ms/update and projected 1.18 hours;
FP16 sampling therefore provides a small additional speedup without changing
critic or optimization precision. The benchmark explicitly performs ten
post-compile updates before timing so `reduce-overhead` CUDA Graph recording is
not incorrectly charged to every update. MuJoCo stepping, logging, and
checkpoint I/O were not measured because that Python environment did not
contain Gym or MuJoCo.

## Packed quantile-ensemble follow-up

The quantile critic was subsequently changed from forty separate per-network
Linear parameters to eight layer-wise packed ensemble parameters. This lets
forward and backward use one batched GEMM per layer and also reduces compile
guards and graph inputs. Legacy `nets.*` checkpoints are migrated on load.

Paired measurements on the same RTX 4090 while the server was under background
load gave:

- previous critic: 31.222 ms per complete update
- packed critic: 24.315 ms and 22.741 ms per complete update
- improvement: approximately **22%–27%**
- behavior/evaluation action: unchanged at approximately 7.25 ms
- compile/CUDA Graph warm-up: reduced from 39.73 s to 22.53–23.95 s
- runtime fullgraphs: still 5
- remote tests: 7 passed

The packed implementation's standalone workload measured 7.175 ms eager and
5.328 ms compiled. The smaller apparent compile-only ratio (1.35x) is expected:
the packed eager implementation is itself substantially more efficient than
the former Python ensemble.
