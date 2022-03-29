import pdb
import traceback

import torch
import os

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.SDE import sde_lib
from model.SDE.ddpm import DDPM
from runners.utils import make_dirs, get_dataset, get_optimizer

from tqdm.autonotebook import tqdm
from PIL import Image


class SDERunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config

        self.args.image_path, self.args.model_path, self.args.log_path, self.args.sample_path, \
        self.args.now = make_dirs(self.args, 'SDE_DDPM', 'test')

    def train(self):
        writer = SummaryWriter(self.args.log_path)
        train_dataset, test_dataset = get_dataset(self.config.data)
        train_loader = DataLoader(train_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                  num_workers=8, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 num_workers=8, drop_last=True)

        # Setup SDEs
        if self.config.training.sde.lower() == 'vpsde':
            sde = sde_lib.VPSDE(beta_min=self.config.model.beta_min, beta_max=self.config.model.beta_max,
                                N=self.config.model.num_scales)
            sampling_eps = 1e-3
        elif self.config.training.sde.lower() == 'subvpsde':
            sde = sde_lib.subVPSDE(beta_min=self.config.model.beta_min, beta_max=self.config.model.beta_max,
                                   N=self.config.model.num_scales)
            sampling_eps = 1e-3
        elif self.config.training.sde.lower() == 'vesde':
            sde = sde_lib.VESDE(sigma_min=self.config.model.sigma_min, sigma_max=self.config.model.sigma_max,
                                N=self.config.model.num_scales)
            sampling_eps = 1e-5
        else:
            raise NotImplementedError(f"SDE {self.config.training.sde} unknown.")

        ddpm_model = DDPM(self.config).to(self.config.device)

        optimizer = get_optimizer(self.config.optimizer, ddpm_model.parameters())
        if self.args.load_model:
            states = torch.load(os.path.join(self.args.model_path, 'checkpoint.pth'))
            ddpm_model.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2000,
                                                               verbose=True, threshold=0.002, threshold_mode='rel',
                                                               cooldown=2000)

        eps = 1e-5
        step = self.args.load_iter
        pbar = tqdm(range(self.config.training.n_epochs), initial=0, dynamic_ncols=True, smoothing=0.01)
        for epoch in pbar:
            for i, (X, y) in enumerate(train_loader):
                try:
                    if step >= self.config.training.n_iters:
                        return 0
                    step += 1
                    ddpm_model.train()
                    X = X.to(self.config.device)
                    X = X / 256. * 255. + torch.rand_like(X) / 256.

                    t = torch.rand(X.shape[0], device=X.device) * (sde.T - eps) + eps
                    z = torch.randn_like(X)
                    mean, std = sde.marginal_prob(X, t)
                    perturbed_data = mean + std[:, None, None, None] * z
                    score = ddpm_model(perturbed_data, t)

                    loss = torch.square(score * std[:, None, None, None] + z)
                    loss = torch.mean(torch.mean(loss.reshape(loss.shape[0], -1), dim=-1))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step(loss)
                    pbar.set_description(
                        (
                            f'iter: {step} loss: {loss:.4f}'
                        )
                    )

                    writer.add_scalar('loss', loss, step)

                except BaseException as e:
                    print('Exception save model start!!!')
                    states = [
                        ddpm_model.state_dict(),
                        optimizer.state_dict(),
                    ]
                    torch.save(states, os.path.join(self.args.model_path, 'checkpoint_exception.pth'))
                    print('Exception save model success!!!')
                    print('str(Exception):\t', str(Exception))
                    print('str(e):\t\t', str(e))
                    print('repr(e):\t', repr(e))
                    print('traceback.print_exc():')
                    traceback.print_exc()
                    print('traceback.format_exc():\n%s' % traceback.format_exc())

















