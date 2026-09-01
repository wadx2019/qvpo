import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    def truncated_normal_init(t, mean=0.0, std=0.01):
        torch.nn.init.normal_(t, mean=mean, std=std)
        while True:
            cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)
        return t

    if type(m) == nn.Linear:
        input_dim = m.in_features
        truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(input_dim)))
        m.bias.data.fill_(0.0)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        scale = math.log(10000) / (half_dim - 1)
        self.register_buffer(
            'frequencies',
            torch.exp(torch.arange(half_dim, dtype=torch.float32) * -scale),
            persistent=False,
        )

    def forward(self, x):
        emb = x[:, None] * self.frequencies[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def quantile_huber_loss(quantiles, samples):
    """Quantile-regression Huber loss used by the distributional critic."""
    pairwise_delta = samples[:, None, None, :] - quantiles[:, :, :, None]
    abs_delta = pairwise_delta.abs()
    huber = torch.where(abs_delta > 1.0,
                        abs_delta - 0.5,
                        0.5 * pairwise_delta.square())
    n_quantiles = quantiles.shape[2]
    tau = ((torch.arange(n_quantiles, device=quantiles.device,
                         dtype=quantiles.dtype) + 0.5) / n_quantiles)
    weight = (tau[None, None, :, None] -
              (pairwise_delta < 0).to(quantiles.dtype)).abs()
    return (weight * huber).mean()

#-----------------------------------------------------------------------------#
#---------------------------------- sampling ---------------------------------#
#-----------------------------------------------------------------------------#


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2, dtype=torch.float32):
    betas = np.linspace(
        beta_start, beta_end, timesteps
    )
    return torch.tensor(betas, dtype=dtype)


def vp_beta_schedule(timesteps, dtype=torch.float32):
    t = np.arange(1, timesteps + 1)
    T = timesteps
    b_max = 10.
    b_min = 0.1
    alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    betas = 1 - alpha
    return torch.tensor(betas, dtype=dtype)

#-----------------------------------------------------------------------------#
#---------------------------------- losses -----------------------------------#
#-----------------------------------------------------------------------------#

class WeightedLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, targ, weights=1.0):
        '''
            pred, targ : tensor [ batch_size x action_dim ]
        '''
        loss = self._loss(pred, targ)
        weighted_loss = (loss * weights).mean()
        return weighted_loss

class WeightedL1(WeightedLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)

class WeightedL2(WeightedLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')


Losses = {
    'l1': WeightedL1,
    'l2': WeightedL2,
}


class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new
