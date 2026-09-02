# [NeurIPS 2024] Diffusion-based Reinforcement Learning via Q-weighted Variational Policy Optimization

Code release for **Diffusion-based Reinforcement Learning via Q-weighted Variational Policy Optimization (NeurIPS 2024)**.

## Full-graph compilation and distributional critic

Training now defaults to the TQC-style distributional quantile critic from the
`online` implementation and enables `torch.compile` with `fullgraph=True`,
static shapes, and the `reduce-overhead` Inductor mode.

The five quantile networks are stored as eight layer-wise packed parameters
instead of forty independent Linear parameters. Their forward and backward
passes therefore use four batched GEMMs. Checkpoints from the previous
`nets.*` layout are migrated automatically when loaded.

Weighted training and 64-candidate action augmentation are always enabled by
default. The old action-gradient and augmentation-gradient paths, their second
replay buffer, and the `--aug` switch have been removed.

The hot tensor path has four runtime graphs: the full online interaction graph,
a critic-training graph containing the complete target DDPM loop, candidate
ranking and quantile loss, an actor-training graph containing its complete DDPM
loop, augmentation ranking/weighting and actor loss, plus critic inference for
utility calls. Target candidate quantiles are gathered directly, avoiding a
selected-action materialization and a second target-critic forward. Explicit
parameter tuples let online and target networks reuse graphs instead of
compiling duplicates. No-grad DDPM sampling defaults to FP16 tensor cores
(`--sampling_precision fp32` disables this), while critic, loss, and optimizer
math remains FP32. Network optimizers use fused Adam and target updates use
foreach kernels.

```bash
# Compiled quantile-critic training (new default)
python main.py --env_name Hopper-v3

# Eager baseline or original twin-Q critic
python main.py --no_compile
python main.py --critic_type double

```

Inductor CUDA compilation requires a working Triton installation. Official
PyTorch Linux CUDA wheels install a matching build. On Windows, install a
`triton-windows` build matching the installed PyTorch, or use `--no_compile`.
An unavailable backend or graph break raises an error instead of silently
falling back to eager execution.

[[paper]](https://arxiv.org/abs/2405.16173) [[project page]](https://dingsht.tech/qvpo-webpage/)

![](./asset/qvpo.png)

## Requirements
Installations of [PyTorch](https://pytorch.org/) and [MuJoCo](https://github.com/deepmind/mujoco) are needed. 
A suitable [conda](https://conda.io) environment named `qvpo` can be created and activated with:
```
conda create qvpo
conda activate qvpo
```
To get started, install the additionally required python packages into your environment.
```
pip install setuptools==65.0.0 wheel==0.38.4
pip install -r requirements.txt
```

## Running
Running experiments based on our code could be quite easy, so below we use `HalfCheetah-v3` task as an example. 

```
python main.py --env_name HalfCheetah-v3
```

## Citation
If you find this repository useful in your research, please consider citing:

```
@inproceedings{
ding2024diffusionbased,
title={Diffusion-based Reinforcement Learning via Q-weighted Variational Policy Optimization},
author={Shutong Ding and Ke Hu and Zhenhao Zhang and Kan Ren and Weinan Zhang and Jingyi Yu, Jingya Wang and Ye Shi},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://arxiv.org/abs/2405.16173}
}
```

## Acknowledgement

The code of QVPO is based on the implementation of [DIPO](https://github.com/BellmanTimeHut/DIPO).
