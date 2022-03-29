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
        dim_mults=(1, 2, 4),
        channels = 1,
        with_time_emb = True
    ):
        super().__init__()
        self.channels = channels

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
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
            ConvNextBlock(dim, dim),
            nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, y, time):
        # pdb.set_trace()
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        x = x + y
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


class CDiffuseNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = self.config.data.image_size
        self.num_timesteps = self.config.model.n_steps
        self.beta_min = self.config.model.beta_min
        self.beta_max = self.config.model.beta_max
        self.loss_type = 'l1'
        self.channels = self.config.data.channels

        self.denoise_fn = CondUnet(dim=config.model.dim, dim_mults=config.model.dim_mults, channels=self.channels)
        # self.denoise_fn = Unet(dim=config.model.dim)

        # betas = cosine_beta_schedule(config.model.n_steps)
        # pdb.set_trace()
        # betas = torch.linspace(self.beta_min, self.beta_max, self.num_timesteps)
        betas = np.linspace(self.beta_min, self.beta_max, self.num_timesteps)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        to_torch = partial(torch.tensor, dtype=torch.float32)

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
        alphas_cumprod_t = extract(self.alphas_cumprod, t, x.shape)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        m_t = sqrt_one_minus_alphas_cumprod_t / torch.sqrt(sqrt_alphas_cumprod_t)
        sigma2_t = 1. - alphas_cumprod_t * (1. + m_t**2)

        alphas_cumprod_tminus = extract(self.alphas_cumprod, tminus, x.shape)
        sqrt_alphas_cumprod_tminus = extract(self.sqrt_alphas_cumprod, tminus, x.shape)
        sqrt_one_minus_alphas_cumprod_tminus = extract(self.sqrt_one_minus_alphas_cumprod, tminus, x.shape)
        m_tminus = sqrt_one_minus_alphas_cumprod_tminus / torch.sqrt(sqrt_alphas_cumprod_tminus)
        sigma2_tminus = 1. - alphas_cumprod_tminus * (1. + m_tminus**2)

        sigma2_tminus_t = sigma2_t - ((1. - m_t) / (1. - m_tminus))**2 * alphas_t * sigma2_tminus

        cxt = ((1. - m_t) * sigma2_tminus * torch.sqrt(alphas_t)) / ((1. - m_tminus) * sigma2_t) + \
              ((1. - m_tminus) * sigma2_tminus_t) / (sigma2_t * torch.sqrt(alphas_t))

        cyt = (m_tminus * sigma2_t - (m_t * (1. - m_t) * alphas_t * sigma2_tminus) / (1. - m_tminus)) * \
              (sqrt_alphas_cumprod_tminus / sigma2_t)

        cet = ((1. - m_tminus) * sigma2_tminus_t * sqrt_one_minus_alphas_cumprod_t) / (sigma2_t * torch.sqrt(alphas_t))

        sigma2_noise = sigma2_tminus_t * sigma2_t / sigma2_tminus

        return cxt * x + cyt * y - cet * self.denoise_fn(x, y, tminus) + \
               torch.sqrt(sigma2_noise) * torch.randn(x.shape, device=x.device)

    @torch.no_grad()
    def p_sample_loop(self, y_start):
        device = self.config.device
        shape = y_start.shape
        b = shape[0]

        noise = torch.randn(shape, device=device)
        T = torch.full((b,), self.num_timesteps - 1, device=device, dtype=torch.long)
        alphas_T = extract(self.alphas_cumprod, T, shape)
        sqrt_alphas_T = extract(self.sqrt_alphas_cumprod, T, shape)
        sigma_T = torch.sqrt((1. - alphas_T) * (1. - sqrt_alphas_T))
        img = sqrt_alphas_T * y_start + noise * sigma_T

        imgs = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps - 1)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, y_start, i)
            imgs.append(img)
        return imgs

    def q_sample(self, x_start, y_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        alphas = extract(self.alphas_cumprod, t, x_start.shape)
        sqrt_alphas = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        m_t = sqrt_one_minus_alphas / torch.sqrt(sqrt_alphas)
        sigma_t = torch.sqrt(1. - (1. + m_t**2) * alphas)

        return (
            sqrt_alphas * ((1. - m_t) * x_start + m_t * y_start) + sigma_t * noise,
            (m_t * sqrt_alphas * (y_start - x_start) + sigma_t * noise) / sqrt_one_minus_alphas
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