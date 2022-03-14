import pdb

import torch.nn as nn
import torch
import numpy as np

PI = torch.from_numpy(np.asarray(np.pi))
EPS = 1.e-7


def log_normal_diag(x, mu, log_var, reduction=None, dim=None):
    log_p = -0.5 * torch.log(2. * PI) - 0.5 * log_var - 0.5 * torch.exp(-log_var) * (x - mu)**2
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p


def log_standard_normal(x, reduction=None, dim=None):
    log_p = -0.5 * torch.log(2. * PI) - 0.5 * x**2
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p


def gaussian_KL(mu_posterior, sigma_posterior, mu, sigma):
    KL = torch.log(sigma) - torch.log(sigma_posterior) + \
         (sigma_posterior**2 + (mu_posterior - mu)**2) / (2*sigma**2) - 0.5
    return KL


class DDGMNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.in_dim = config.data.channels * config.data.image_size * config.data.image_size
        self.mid_dim = config.model.mid_dim
        self.P_DNNS = nn.ModuleList([nn.Sequential(nn.Linear(self.in_dim, self.mid_dim), nn.LeakyReLU(),
                                     nn.Linear(self.mid_dim, self.mid_dim), nn.LeakyReLU(),
                                     nn.Linear(self.mid_dim, self.mid_dim), nn.LeakyReLU(),
                                     nn.Linear(self.mid_dim, self.in_dim * 2)) for _ in range(config.model.n_steps - 1)])
        self.decoder_net = nn.Sequential(nn.Linear(self.in_dim, self.mid_dim * 2), nn.LeakyReLU(),
                                         nn.Linear(self.mid_dim * 2, self.mid_dim * 2), nn.LeakyReLU(),
                                         nn.Linear(self.mid_dim * 2, self.mid_dim * 2), nn.LeakyReLU(),
                                         nn.Linear(self.mid_dim * 2, self.in_dim), nn.Tanh())
        self.betas = self.make_beta_schedule(config.model.schedule, config.model.n_steps,
                                             config.model.sigma_begin, config.model.sigma_end).to(config.device)

    def make_beta_schedule(self, schedule='linear', n_steps=1000, start=1e-5, end=1e-2):
        betas = None
        if schedule == 'linear':
            betas = torch.linspace(start, end, n_steps)
        elif schedule == "quad":
            betas = torch.linspace(start ** 0.5, end ** 0.5, n_steps) ** 2
        elif schedule == "sigmoid":
            betas = torch.linspace(-6, 6, n_steps)
            betas = torch.sigmoid(betas) * (end - start) + start
        return betas

    def reparameterization_gaussian_diffusion(self, x, i):
        return torch.sqrt(1 - self.betas[i]) * x + torch.sqrt(self.betas[i]) * torch.randn_like(x)

    @torch.no_grad()
    def sample(self, noise):
        noise = noise.view(noise.shape[0], -1)
        # backward diffusion
        mus = [noise]

        for i in range(len(self.P_DNNS) - 1, -1, -1):
            h = self.P_DNNS[i](mus[-1])
            mu_i, _ = torch.chunk(h, 2, dim=1)
            mus.append(mu_i)

        mu_x = self.decoder_net(mus[-1])
        mus.append(mu_x)
        return mus

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        # forward diffusion
        zs = [self.reparameterization_gaussian_diffusion(x, 0)]
        for i in range(1, len(self.betas)):
            zs.append(self.reparameterization_gaussian_diffusion(zs[-1], i))

        # backward diffusion
        mus = []
        log_vars = []

        # pdb.set_trace()
        for i in range(len(self.P_DNNS) - 1, -1, -1):
            h = self.P_DNNS[i](zs[i + 1])
            mu_i, log_var_i = torch.chunk(h, 2, dim=1)
            mus.insert(0, mu_i)
            log_vars.insert(0, log_var_i)

        mu_x = self.decoder_net(zs[0])
        # pdb.set_trace()

        # ELBO
        # RE:Reconstruction
        RE = log_standard_normal(x - mu_x).mean(-1)

        # KL:KL Divergence
        KL = gaussian_KL(zs[-1], torch.sqrt(self.betas[-1]), torch.zeros_like(zs[-1]),
                         torch.ones_like(self.betas[-1])).mean(-1)

        for i in range(len(mus)):
            # print(f'{zs[i]}, {torch.sqrt(self.betas[i])}, {mus[i]}, {torch.sqrt(vars[i])}')
            KL_i = gaussian_KL(zs[i], torch.sqrt(self.betas[i]), mus[i], torch.sqrt(torch.exp(log_vars[i]))).mean(-1)
            # print(KL_i)
            KL = KL + KL_i

        # print(KL)
        # Final ELBO
        loss = -(RE - KL).mean()

        zs.insert(0, x)
        mus.insert(0, mu_x)
        return loss, zs, mus