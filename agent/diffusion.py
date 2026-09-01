import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from agent.helpers import (cosine_beta_schedule,
                            linear_beta_schedule,
                            vp_beta_schedule,
                            extract,
                            Losses)

from agent.model import Model


class Diffusion(nn.Module):
    def __init__(self, state_dim, action_dim, noise_ratio,
                 beta_schedule='vp', n_timesteps=1000,
                 loss_type='l2', clip_denoised=True, predict_epsilon=True,
                 behavior_sample=16, eval_sample=512, deterministic=False):
        super(Diffusion, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = Model(state_dim, action_dim)
        self._compiled_model_forward = None
        self._compiled_sample_step = None
        self._compiled_sample_loop = None
        self.sampling_dtype = None

        self.max_noise_ratio = noise_ratio
        self.noise_ratio = noise_ratio

        self.behavior_sample = behavior_sample
        self.eval_sample = eval_sample
        self.deterministic = deterministic

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(n_timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(n_timesteps)
        elif beta_schedule == 'vp':
            betas = vp_beta_schedule(n_timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        self.loss_fn = Losses[loss_type]()

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, s):
        if self._compiled_model_forward is None:
            model_output = self.model(x, t, s)
        else:
            model_output = self._compiled_model_forward(
                x, t, s, tuple(self.model.parameters())
            )
        x_recon = self.predict_start_from_noise(x, t=t, noise=model_output)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def set_compiled_model_forward(self, compiled_forward):
        """Install the shared full-graph denoiser callable after deepcopy."""
        self._compiled_model_forward = compiled_forward

    def set_compiled_sample_step(self, compiled_step):
        """Install the graph that fuses a complete DDPM sampling step."""
        self._compiled_sample_step = compiled_step

    def set_compiled_sample_loop(self, compiled_sample_loop):
        """Install one graph containing the complete fixed-step DDPM loop."""
        self._compiled_sample_loop = compiled_sample_loop

    def functional_sample_loop(self, state, x, parameters, noise_ratio):
        """Stateless, fully unrolled sampler shared by actor and target actor."""
        batch_size = x.shape[0]
        for i in reversed(range(self.n_timesteps)):
            timesteps = torch.full(
                (batch_size,), i, device=x.device, dtype=torch.long
            )
            model_output = self.model.functional_forward(
                x, timesteps, state, parameters
            )
            if self.predict_epsilon:
                x_recon = (
                    self.sqrt_recip_alphas_cumprod[i] * x
                    - self.sqrt_recipm1_alphas_cumprod[i] * model_output
                )
            else:
                x_recon = model_output
            if self.clip_denoised:
                x_recon = x_recon.clamp(-1.0, 1.0)

            model_mean = (
                self.posterior_mean_coef1[i] * x_recon
                + self.posterior_mean_coef2[i] * x
            )
            noise = torch.randn_like(x)
            if i > 0:
                x = (
                    model_mean
                    + (0.5 * self.posterior_log_variance_clipped[i]).exp()
                    * noise * noise_ratio
                )
            else:
                x = model_mean
        return x

    def functional_sample_step(self, x, t, state, parameters, noise_ratio):
        """Stateless denoiser, posterior, and noise update for one DDPM step."""
        use_amp = self.sampling_dtype is not None and x.device.type == 'cuda'
        amp_dtype = self.sampling_dtype or torch.float16
        with torch.autocast(
            device_type=x.device.type, dtype=amp_dtype, enabled=use_amp
        ):
            model_output = self.model.functional_forward(
                x, t, state, parameters
            )
        if self.predict_epsilon:
            x_recon = (
                extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
                * model_output
            )
        else:
            x_recon = model_output
        if self.clip_denoised:
            x_recon = x_recon.clamp(-1.0, 1.0)
        model_mean = (
            extract(self.posterior_mean_coef1, t, x.shape) * x_recon
            + extract(self.posterior_mean_coef2, t, x.shape) * x
        )
        model_log_variance = extract(
            self.posterior_log_variance_clipped, t, x.shape
        )
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).to(x.dtype).reshape(x.shape[0], 1)
        return (
            model_mean
            + nonzero_mask * (0.5 * model_log_variance).exp()
            * noise * noise_ratio
        )

    @torch.no_grad()
    def p_sample(self, x, t, s):
        if self._compiled_sample_step is not None:
            return self._compiled_sample_step(
                x, t, s, tuple(self.model.parameters()), self.noise_ratio
            ).clone()
        b, *_, device = *x.shape, x.device

        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s)

        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise * self.noise_ratio


    @torch.no_grad()
    def p_sample_loop(self, state, shape):
        device = self.betas.device

        x = torch.randn(shape, device=device)

        if self._compiled_sample_loop is not None:
            # The result feeds a different compiled graph (critic ranking).
            # Clone outside the CUDA Graph-owned storage before that call.
            return self._compiled_sample_loop(
                state, x, tuple(self.model.parameters()), self.noise_ratio
            ).clone()

        batch_size = shape[0]
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps, state)

        return x

    @torch.no_grad()
    def sample(self, state, eval=False, q_func=None, normal=False):
        if self.deterministic:
            self.noise_ratio = 0 if eval else self.max_noise_ratio
        else:
            self.noise_ratio = self.max_noise_ratio

        if normal:
            batch_size = state.shape[0]
            shape = (batch_size, self.action_dim)
            action = self.p_sample_loop(state, shape)
            action.clamp_(-1., 1.)
            return action

        if eval:
            raw_batch_size = state.shape[0]
            state = state.repeat(self.eval_sample, 1)
            batch_size = state.shape[0]
            shape = (batch_size, self.action_dim)
            action = self.p_sample_loop(state, shape)
            action.clamp_(-1., 1.)
            q_output = q_func(state, action)
            q = (torch.min(q_output[0], q_output[1])
                 if isinstance(q_output, tuple) else q_output)
            action = action.view(self.eval_sample, raw_batch_size, -1).transpose(0,1)
            q = q.view(self.eval_sample, raw_batch_size, -1).transpose(0,1)
            action_idx = torch.argmax(q, dim=1, keepdim=True).repeat(1,1,self.action_dim)
            return action.gather(dim=1, index=action_idx).view(raw_batch_size, -1)
        else:
            raw_batch_size = state.shape[0]
            state = state.repeat(self.behavior_sample, 1)
            batch_size = state.shape[0]
            shape = (batch_size, self.action_dim)
            action = self.p_sample_loop(state, shape)
            action.clamp_(-1., 1.)
            q_output = q_func(state, action)
            q = (torch.min(q_output[0], q_output[1])
                 if isinstance(q_output, tuple) else q_output)
            action = action.view(self.behavior_sample, raw_batch_size, -1).transpose(0,1)
            q = q.view(self.behavior_sample, raw_batch_size, -1).transpose(0,1)
            action_idx = torch.argmax(q, dim=1, keepdim=True).repeat(1,1,self.action_dim)
            return action.gather(dim=1, index=action_idx).view(raw_batch_size, -1)

    # ------------------------------------------ training ------------------------------------------#
    @torch.no_grad()
    def sample_n(self, state, eval=False, times=32, chosen=1, q_func=None, origin=None):
        self.noise_ratio = self.max_noise_ratio
        old_state = state
        # q1, q2 = q_func(state, origin)
        # q_origin = 0.9 * torch.min(q1, q2) + 0.1 * torch.max(q1, q2)
        raw_batch_size = state.shape[0]
        state = state.repeat(times, 1)
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        action = self.p_sample_loop(state, shape)
        action.clamp_(-1., 1.)
        q_output = q_func(state, action)
        q = (torch.min(q_output[0], q_output[1])
             if isinstance(q_output, tuple) else q_output)
        action = action.view(times, raw_batch_size, -1).transpose(0, 1)
        q = q.view(times, raw_batch_size, -1).transpose(0, 1)
        mean = q.mean()
        std = q.std()
        v = q.mean(dim=1, keepdim=True)

        # q_prox = torch.linalg.norm(action - origin.view(raw_batch_size, 1, -1), dim=-1, keepdim=True)
        q_prox = q #- 0.0 / math.sqrt(self.action_dim) * q.std(dim=1, keepdim=True) * q_prox

        if chosen == 1:
            _, q_idx = torch.max(q_prox, dim=1, keepdim=True)
            action_idx = q_idx.repeat(1, 1, self.action_dim)
            q = q.gather(dim=1, index=q_idx).view(raw_batch_size, 1)
            # q = torch.where(q>q_origin, q, v.view(raw_batch_size, 1))

            return old_state, action.gather(dim=1, index=action_idx).view(raw_batch_size, -1), (q.view(raw_batch_size, 1), v), (mean, std)
        else:
            q, q_idx = torch.topk(q, k=chosen, dim=1)
            action_idx = q_idx.repeat(1, 1, self.action_dim)
            return old_state.repeat(chosen, 1).view(chosen, raw_batch_size, -1).transpose(0,1).contiguous().view(raw_batch_size*chosen, -1), action.gather(dim=1, index=action_idx).view(raw_batch_size*chosen, -1), (q.view(raw_batch_size*chosen, 1), v), (mean, std)


    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, state, t, weights=1.0):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_recon = self.model(x_noisy, t, state)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise, weights)
        else:
            loss = self.loss_fn(x_recon, x_start, weights)

        return loss


    def loss(self, x, state, weights=1.0):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, state, t, weights)

    def forward(self, state, eval=False, q_func=None, normal=False):
        return self.sample(state, eval, q_func, normal)

