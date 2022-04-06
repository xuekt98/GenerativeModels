import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from tqdm.autonotebook import tqdm
import numpy as np

from model.DDPM.ddpmnet import default, cosine_beta_schedule, extract, SinusoidalPosEmb, ConvNextBlock, Residual, \
    PreNorm, LinearAttention, Downsample, Upsample, exists


class CondUnet(nn.Module):
    def __init__(
        self,
        dim,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 1,
        with_time_emb = True
    ):
        super().__init__()
        self.channels = channels

        dims = [channels * 2, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ConvNextBlock(dim_in, dim_out, time_emb_dim = time_dim, norm = ind != 0),
                ConvNextBlock(dim_out, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ConvNextBlock(dim_out * 2, dim_in, time_emb_dim = time_dim),
                ConvNextBlock(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            ConvNextBlock(dim, 2*dim),
            Residual(PreNorm(2*dim, LinearAttention(2*dim))),
            # ConvNextBlock(2*dim, 4*dim),
            # Residual(PreNorm(4*dim, LinearAttention(4*dim))),
            # ConvNextBlock(4*dim, 2*dim),
            # Residual(PreNorm(2*dim, LinearAttention(2*dim))),
            ConvNextBlock(2*dim, dim),
            nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, y, time):
        # pdb.set_trace()
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []
        x = torch.cat((x, y), dim=1)
        # pdb.set_trace()
        for convnext, convnext2, attn, downsample in self.downs:
            x = convnext(x, t)
            x = convnext2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # pdb.set_trace()
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for convnext, convnext2, attn, upsample in self.ups:
            # pdb.set_trace()
            x = torch.cat((x, h.pop()), dim=1)
            x = convnext(x, t)
            x = convnext2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)


class BrownianBridgeNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = self.config.data.image_size
        self.num_timesteps = self.config.model.n_steps
        self.beta_min = self.config.model.beta_min
        self.beta_max = self.config.model.beta_max
        self.loss_type = self.config.model.loss_type
        self.channels = self.config.data.channels

        self.denoise_fn = CondUnet(dim=config.model.dim, dim_mults=config.model.dim_mults, channels=self.channels)
        # self.denoise_fn = Unet(dim=config.model.dim)

        # betas = cosine_beta_schedule(config.model.n_steps)
        # betas = np.linspace(self.beta_min, self.beta_max, self.num_timesteps)
        ms, variance = None, None
        T = self.num_timesteps - 1
        if self.config.model.mt_type == 'linear': ## linear mt
            ms = np.arange(self.num_timesteps)
            ms = ms / T
        elif self.config.model.mt_type == 'sin': ## sin mt
            ms = np.arange(self.num_timesteps)
            ms = ms / T
            ms = 0.5 * np.sin(np.pi * (ms - 0.5)) + 0.5

        if self.config.model.var_type == 'quard':
            variance = np.arange(self.num_timesteps)
            variance = variance * (T - variance) / T
            variance = variance * self.beta_max * 4. / T + self.beta_min
        elif self.config.model.var_type == 'sin':
            variance = np.arange(self.num_timesteps)
            variance = variance / T
            variance = (0.5 * np.sin(np.pi * (2. * variance - 0.5)) + 0.5) * self.beta_max + self.beta_min

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('variance', to_torch(variance))
        self.register_buffer('ms', to_torch(ms))

    @torch.no_grad()
    def p_sample(self, x, y, i):
        b = x.shape[0]
        t = torch.full((b,), i, device=x.device, dtype=torch.long)
        tminus = torch.full((b,), i-1, device=x.device, dtype=torch.long)

        m_t = extract(self.ms, t, x.shape)
        sigma2_t = extract(self.variance, t, x.shape)

        m_tminus = extract(self.ms, tminus, x.shape)
        sigma2_tminus = extract(self.variance, tminus, x.shape)

        sigma2_tminus_t = sigma2_t - ((1. - m_t) / (1. - m_tminus))**2 * sigma2_tminus

        cxt = ((1. - m_t) * sigma2_tminus) / ((1. - m_tminus) * sigma2_t) + ((1. - m_tminus) * sigma2_tminus_t) / sigma2_t

        cyt = (m_tminus * sigma2_t - (m_t * (1. - m_t) * sigma2_tminus) / (1. - m_tminus)) / sigma2_t

        cet = ((1. - m_tminus) * sigma2_tminus_t) / sigma2_t

        sigma2_noise = sigma2_tminus_t * sigma2_tminus / sigma2_t
        # pdb.set_trace()
        # if i < self.num_timesteps - 1:
        #     pdb.set_trace()
        if i == 1:
            return cxt * x + cyt * y - cet * self.denoise_fn(x, y, t)
        else:
            return cxt * x + cyt * y - cet * self.denoise_fn(x, y, t) + \
                torch.sqrt(sigma2_noise) * torch.randn(x.shape, device=x.device)

    @torch.no_grad()
    def p_sample_loop(self, y_start, x_start=None, step=None):
        if step is None:
            step = self.num_timesteps
            t = torch.full((y_start.shape[0],), step - 1, device=y_start.device, dtype=torch.long)
            var_t = extract(self.variance, t, y_start.shape)

            noise = torch.randn_like(y_start)
            img = y_start + torch.sqrt(var_t) * noise
        else:
            t = torch.full((y_start.shape[0],), step, device=y_start.device, dtype=torch.long)
            img = self.q_sample(x_start, y_start, t)

        imgs = [img]
        for i in tqdm(reversed(range(1, step)), desc='p sampling loop time step', total=step):
            img = self.p_sample(img, y_start, i)
            imgs.append(img)
        return imgs

    @torch.no_grad()
    def q_sample_loop(self, x_start, y_start):
        imgs = []
        for i in tqdm(range(self.num_timesteps), desc="q sampling loop", total=self.num_timesteps):
            t = torch.full((y_start.shape[0],), i, device=x_start.device, dtype=torch.long)
            img, _ = self.q_sample(x_start, y_start, t)
            imgs.append(img)
        return imgs

    def q_sample(self, x_start, y_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        m_t = extract(self.ms, t, x_start.shape)
        var_t = extract(self.variance, t, x_start.shape)
        sigma_t = torch.sqrt(var_t)

        return (
            (1. - m_t) * x_start + m_t * y_start + sigma_t * noise,
            m_t * (y_start - x_start) + sigma_t * noise
        )

    def p_losses(self, x_start, y_start, t, noise=None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        xy_noisy, grad = self.q_sample(x_start=x_start, y_start=y_start, t=t, noise=noise)

        # pdb.set_trace()
        xy_recon = self.denoise_fn(xy_noisy, y_start, t)

        if self.loss_type == 'l1':
            loss = (grad - xy_recon).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(grad, xy_recon)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, y):
        b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, y, t)


class BrownianBridgeNet_2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = self.config.data.image_size
        self.num_timesteps = self.config.model.n_steps
        self.beta_min = self.config.model.beta_min
        self.beta_max = self.config.model.beta_max
        self.loss_type = self.config.model.loss_type
        self.channels = self.config.data.channels

        self.denoise_fn = CondUnet(dim=config.model.dim, dim_mults=config.model.dim_mults, channels=self.channels)
        # self.denoise_fn = Unet(dim=config.model.dim)

        # betas = cosine_beta_schedule(config.model.n_steps)
        # betas = np.linspace(self.beta_min, self.beta_max, self.num_timesteps)
        ms, variance = None, None
        T = self.num_timesteps - 1
        if self.config.model.mt_type == 'linear': ## linear mt
            ms = np.arange(self.num_timesteps)
            ms = ms / T
        elif self.config.model.mt_type == 'sin': ## sin mt
            ms = np.arange(self.num_timesteps)
            ms = ms / T
            ms = 0.5 * np.sin(np.pi * (ms - 0.5)) + 0.5
        ms[-1] = 1.00001

        if self.config.model.var_type == 'quard':
            variance = np.arange(self.num_timesteps)
            variance = variance * (T - variance) / T
            variance = variance * self.beta_max * 4. / T + self.beta_min
        elif self.config.model.var_type == 'sin':
            variance = np.arange(self.num_timesteps)
            variance = variance / T
            variance = (0.5 * np.sin(np.pi * (2. * variance - 0.5)) + 0.5) * self.beta_max + self.beta_min

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('variance', to_torch(variance))
        self.register_buffer('ms', to_torch(ms))

    @torch.no_grad()
    def p_sample(self, x, y, i):
        b = x.shape[0]
        t = torch.full((b,), i, device=x.device, dtype=torch.long)
        tminus = torch.full((b,), i-1, device=x.device, dtype=torch.long)

        m_t = extract(self.ms, t, x.shape)
        sigma2_t = extract(self.variance, t, x.shape)

        m_tminus = extract(self.ms, tminus, x.shape)
        sigma2_tminus = extract(self.variance, tminus, x.shape)

        sigma2_tminus_t = sigma2_t - ((1. - m_t) / (1. - m_tminus))**2 * sigma2_tminus

        cxt = (1. - m_tminus) / (1. - m_t)
        cyt = m_tminus - m_t * (1. - m_tminus) / (1. - m_t)
        cet = ((1. - m_tminus) * sigma2_tminus_t) / (torch.sqrt(sigma2_t) * (1. - m_t))

        sigma2_noise = sigma2_tminus_t * sigma2_tminus / sigma2_t
        # pdb.set_trace()
        if i == 1:
            return cxt * x + cyt * y - cet * self.denoise_fn(x, y, t)
        else:
            return cxt * x + cyt * y - cet * self.denoise_fn(x, y, t) + \
                torch.sqrt(sigma2_noise) * torch.randn(x.shape, device=x.device)

    @torch.no_grad()
    def p_sample_loop(self, y_start, x_start=None, step=None):
        if step is None:
            step = self.num_timesteps
            t = torch.full((y_start.shape[0],), step - 1, device=y_start.device, dtype=torch.long)
            var_t = extract(self.variance, t, y_start.shape)

            noise = torch.randn_like(y_start)
            img = y_start + torch.sqrt(var_t) * noise
        else:
            t = torch.full((y_start.shape[0],), step, device=y_start.device, dtype=torch.long)
            img = self.q_sample(x_start, y_start, t)

        imgs = [img]
        for i in tqdm(reversed(range(1, step)), desc='p sampling loop time step', total=step):
            img = self.p_sample(img, y_start, i)
            imgs.append(img)
        return imgs

    @torch.no_grad()
    def q_sample_loop(self, x_start, y_start):
        imgs = []
        for i in tqdm(range(self.num_timesteps), desc="q sampling loop", total=self.num_timesteps):
            t = torch.full((y_start.shape[0],), i, device=x_start.device, dtype=torch.long)
            img, _ = self.q_sample(x_start, y_start, t)
            imgs.append(img)
        return imgs

    def q_sample(self, x_start, y_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        m_t = extract(self.ms, t, x_start.shape)
        var_t = extract(self.variance, t, x_start.shape)
        sigma_t = torch.sqrt(var_t)

        return (
            (1. - m_t) * x_start + m_t * y_start + sigma_t * noise,
            noise
        )

    def p_losses(self, x_start, y_start, t, noise=None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        xy_noisy, grad = self.q_sample(x_start=x_start, y_start=y_start, t=t, noise=noise)

        # pdb.set_trace()
        xy_recon = self.denoise_fn(xy_noisy, y_start, t)

        if self.loss_type == 'l1':
            loss = (grad - xy_recon).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(grad, xy_recon)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, y):
        b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, y, t)


class CDiffuseBrownianBridgeNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = self.config.data.image_size
        self.num_timesteps = self.config.model.n_steps
        self.beta_min = self.config.model.beta_min
        self.beta_max = self.config.model.beta_max
        self.loss_type = self.config.model.loss_type
        self.channels = self.config.data.channels

        self.denoise_fn = CondUnet(dim=config.model.dim, dim_mults=config.model.dim_mults, channels=self.channels)
        # self.denoise_fn = Unet(dim=config.model.dim)

        # betas = cosine_beta_schedule(config.model.n_steps)
        betas = np.linspace(self.beta_min, self.beta_max, self.num_timesteps)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        ms, variance = None, None
        if self.config.model.mt_type == 'aa':
            ms = (1. - alphas_cumprod) / alphas_cumprod
        elif self.config.model.mt_type == 'tTaa':
            tT = np.linspace(0., 1., self.num_timesteps)
            ms = tT * (1. - alphas_cumprod) / alphas_cumprod
        elif self.config.model.mt_type == 'linear':
            ms = np.arange(self.num_timesteps)
            ms = ms / (self.num_timesteps - 1.)
            ms[0] = 0.0001
            ms[-1] = 1.0001
        elif self.config.model.mt_type == 'CD':
            ms = np.sqrt((1. - alphas_cumprod) / np.sqrt(alphas_cumprod))

        if self.config.model.var_type == 'CD':
            variance = 1. - alphas_cumprod - ms ** 2 * alphas_cumprod

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('ms', to_torch(ms))
        self.register_buffer('variance', to_torch(variance))

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas', to_torch(alphas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    @torch.no_grad()
    def p_sample(self, x, y, i):
        b = x.shape[0]
        tminus = torch.full((b,), i, device=x.device, dtype=torch.long)
        t = torch.full((b,), i+1, device=x.device, dtype=torch.long)

        alphas_t = extract(self.alphas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        m_t = extract(self.ms, t, x.shape)
        sigma2_t = extract(self.variance, t, x.shape)

        sqrt_alphas_cumprod_tminus = extract(self.sqrt_alphas_cumprod, tminus, x.shape)
        m_tminus = extract(self.ms, tminus, x.shape)
        sigma2_tminus = extract(self.variance, tminus, x.shape)

        sigma2_tminus_t = sigma2_t - ((1. - m_t) / (1. - m_tminus))**2 * alphas_t * sigma2_tminus

        cxt = ((1. - m_t) * sigma2_tminus * torch.sqrt(alphas_t)) / ((1. - m_tminus) * sigma2_t) + \
              ((1. - m_tminus) * sigma2_tminus_t) / (sigma2_t * torch.sqrt(alphas_t))

        cyt = (m_tminus * sigma2_t - (m_t * (1. - m_t) * alphas_t * sigma2_tminus) / (1. - m_tminus)) * \
              (sqrt_alphas_cumprod_tminus / sigma2_t)

        cet = ((1. - m_tminus) * sigma2_tminus_t * sqrt_one_minus_alphas_cumprod_t) / (sigma2_t * torch.sqrt(alphas_t))

        sigma2_noise = sigma2_tminus_t * sigma2_tminus / sigma2_t

        if i == 0:
            return cxt * x + cyt * y - cet * self.denoise_fn(x, y, t)
        else:
            return cxt * x + cyt * y - cet * self.denoise_fn(x, y, t) + \
                torch.sqrt(sigma2_noise) * torch.randn(x.shape, device=x.device)

    @torch.no_grad()
    def p_sample_loop(self, y_start, x_start=None, step=None):
        if step is None:
            step = self.num_timesteps
            t = torch.full((y_start.shape[0],), step - 1, device=y_start.device, dtype=torch.long)
            sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, y_start.shape)
            var_t = extract(self.variance, t, y_start.shape)

            noise = torch.randn_like(y_start)
            img = sqrt_alphas_cumprod_t * y_start + torch.sqrt(var_t) * noise
        else:
            t = torch.full((y_start.shape[0],), step, device=y_start.device, dtype=torch.long)
            img, _ = self.q_sample(x_start, y_start, t)

        imgs = [img]
        for i in tqdm(reversed(range(0, step - 1)), desc='sampling loop time step', total=step):
            img = self.p_sample(img, y_start, i)
            imgs.append(img)
        return imgs

    def q_sample(self, x_start, y_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        sqrt_alphas = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        m_t = extract(self.ms, t, x_start.shape)
        sigma_t = torch.sqrt(extract(self.variance, t, x_start.shape))

        return (
            sqrt_alphas * ((1. - m_t) * x_start + m_t * y_start) + sigma_t * noise,
            (m_t * sqrt_alphas * (y_start - x_start) + sigma_t * noise) / sqrt_one_minus_alphas
        )

    def p_losses(self, x_start, y_start, t, noise=None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        xy_noisy, grad = self.q_sample(x_start=x_start, y_start=y_start, t=t, noise=noise)

        xy_recon = self.denoise_fn(xy_noisy, y_start, t)

        if self.loss_type == 'l1':
            loss = (grad - xy_recon).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(grad, xy_recon)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, y):
        b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, y, t)


class CDiffuseBrownianBridgeNet_2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = self.config.data.image_size
        self.num_timesteps = self.config.model.n_steps
        self.beta_min = self.config.model.beta_min
        self.beta_max = self.config.model.beta_max
        self.loss_type = self.config.model.loss_type
        self.channels = self.config.data.channels

        self.denoise_fn = CondUnet(dim=config.model.dim, dim_mults=config.model.dim_mults, channels=self.channels)
        # self.denoise_fn = Unet(dim=config.model.dim)

        # betas = cosine_beta_schedule(config.model.n_steps)
        betas = np.linspace(self.beta_min, self.beta_max, self.num_timesteps)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        ms, variance = None, None
        if self.config.model.mt_type == 'aa':
            ms = (1. - alphas_cumprod) / alphas_cumprod
        elif self.config.model.mt_type == 'tTaa':
            tT = np.linspace(0., 1., self.num_timesteps)
            ms = tT * (1. - alphas_cumprod) / alphas_cumprod
        elif self.config.model.mt_type == 'linear':
            ms = np.arange(self.num_timesteps)
            ms = ms / (self.num_timesteps - 1.)
        elif self.config.model.mt_type == 'CD':
            ms = np.sqrt((1. - alphas_cumprod) / np.sqrt(alphas_cumprod))
        ms[-1] = 0.9999

        if self.config.model.var_type == 'CD':
            variance = 1. - alphas_cumprod - ms ** 2 * alphas_cumprod

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('ms', to_torch(ms))
        self.register_buffer('variance', to_torch(variance))

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas', to_torch(alphas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    @torch.no_grad()
    def p_sample(self, x, y, i):
        b = x.shape[0]
        tminus = torch.full((b,), i, device=x.device, dtype=torch.long)
        t = torch.full((b,), i+1, device=x.device, dtype=torch.long)

        alphas_t = extract(self.alphas, t, x.shape)
        m_t = extract(self.ms, t, x.shape)
        sigma2_t = extract(self.variance, t, x.shape)

        sqrt_alphas_cumprod_tminus = extract(self.sqrt_alphas_cumprod, tminus, x.shape)
        m_tminus = extract(self.ms, tminus, x.shape)
        sigma2_tminus = extract(self.variance, tminus, x.shape)

        sigma2_tminus_t = sigma2_t - ((1. - m_t) / (1. - m_tminus))**2 * alphas_t * sigma2_tminus

        cxt = ((1. - m_t) * sigma2_tminus * torch.sqrt(alphas_t)) / ((1. - m_tminus) * sigma2_t) + \
              ((1. - m_tminus) * sigma2_tminus_t) / (sigma2_t * torch.sqrt(alphas_t) * (1. - m_t))

        cyt = ((m_tminus * sigma2_t - (m_t * (1. - m_t) * alphas_t * sigma2_tminus) / (1. - m_tminus)) / sigma2_t -
               ((1. - m_tminus) * sigma2_tminus_t * m_t) / (sigma2_t * (1. - m_t))) * sqrt_alphas_cumprod_tminus

        cet = ((1. - m_tminus) * sigma2_tminus_t) / ((1. - m_t) * torch.sqrt(sigma2_t) * torch.sqrt(alphas_t))

        sigma2_noise = sigma2_tminus_t * sigma2_tminus / sigma2_t

        if i == 0:
            return cxt * x + cyt * y - cet * self.denoise_fn(x, y, t)
        else:
            return cxt * x + cyt * y - cet * self.denoise_fn(x, y, t) + \
                torch.sqrt(sigma2_noise) * torch.randn(x.shape, device=x.device)

    @torch.no_grad()
    def p_sample_loop(self, y_start, x_start=None, step=None):
        if step is None:
            step = self.num_timesteps
            t = torch.full((y_start.shape[0],), step - 1, device=y_start.device, dtype=torch.long)
            sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, y_start.shape)
            var_t = extract(self.variance, t, y_start.shape)

            noise = torch.randn_like(y_start)
            img = sqrt_alphas_cumprod_t * y_start + torch.sqrt(var_t) * noise
        else:
            t = torch.full((y_start.shape[0],), step, device=y_start.device, dtype=torch.long)
            img, _ = self.q_sample(x_start, y_start, t)

        imgs = [img]
        for i in tqdm(reversed(range(0, step - 1)), desc='sampling loop time step', total=step):
            img = self.p_sample(img, y_start, i)
            imgs.append(img)
        return imgs

    def q_sample(self, x_start, y_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        sqrt_alphas_cumprod = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        m_t = extract(self.ms, t, x_start.shape)
        sigma_t = torch.sqrt(extract(self.variance, t, x_start.shape))

        return (
            sqrt_alphas_cumprod * ((1. - m_t) * x_start + m_t * y_start) + sigma_t * noise,
            noise
        )

    def p_losses(self, x_start, y_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        xy_noisy, grad = self.q_sample(x_start=x_start, y_start=y_start, t=t, noise=noise)
        xy_recon = self.denoise_fn(xy_noisy, y_start, t)

        if self.loss_type == 'l1':
            loss = (grad - xy_recon).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(grad, xy_recon)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, y):
        b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, y, t)