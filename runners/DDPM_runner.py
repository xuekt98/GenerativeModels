import torch
import os

from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from runners.utils import get_optimizer, get_dataset, make_dirs, mkdir

from tqdm.autonotebook import tqdm
from PIL import Image


class DDPMRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config

        self.args.image_path, self.args.model_path, self.args.log_path, self.args.sample_path, \
        self.args.now = make_dirs(self.args, 'DDPM', 'test')

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

    def q_sample(self, betas, x_0, n_steps=1000):
        sample_path = os.path.join(self.args.sample_path, 'forward')
        mkdir(sample_path)

        x_mod = x_0
        images = []
        for i in tqdm(range(n_steps), initial=0, desc='forward'):
            # print(f'i={i}, beta[i]={betas[i]:.4f}')
            sample = torch.clamp(x_mod, 0.0, 1.0).to('cpu')
            x_mod = x_mod * torch.sqrt(1 - betas[i]) + torch.randn_like(x_mod) * torch.sqrt(betas[i])
            image_grid = make_grid(sample, nrow=4)
            if i % 10 == 0:
                im = Image.fromarray(
                    image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())

                images.append(im)

            if i % 50 == 0:
                save_image(image_grid, os.path.join(sample_path, 'image_{}.png'.format(i)))

        images[0].save(os.path.join(sample_path, "movie.gif"), save_all=True, append_images=images[1:],
                     duration=1, loop=0)
        return x_mod

    def p_sample(self, betas, x_n=None, n_steps=1000):
        sample_path = os.path.join(self.args.sample_path, 'backward')
        mkdir(sample_path)

        if x_n is None:
            x_n = torch.rand(16, 1, 28, 28, device=self.config.device)
            sample_path = os.path.join(sample_path, "sample")
        else:
            sample_path = os.path.join(sample_path, "rec")
        mkdir(sample_path)

        x_mod = x_n
        images = []
        for i in tqdm(range(n_steps), initial=0, desc='backward'):
            sample = torch.clamp(x_mod, 0.0, 1.0).to('cpu')
            x_mod = (x_mod - torch.randn_like(x_mod) * torch.sqrt(betas[n_steps - i - 1])) / torch.sqrt(1 - betas[n_steps - i - 1])
            image_grid = make_grid(sample, nrow=4)
            if i % 10 == 0:
                im = Image.fromarray(
                    image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())

                images.append(im)

            if i % 50 == 0:
                save_image(image_grid, os.path.join(sample_path, 'image_{}.png'.format(i)))

        images[0].save(os.path.join(sample_path, "movie.gif"), save_all=True, append_images=images[1:],
                       duration=1, loop=0)

    def train(self):
        train_dataset, test_dataset = get_dataset(self.config.data)
        train_loader = DataLoader(train_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                  num_workers=8, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 num_workers=8, drop_last=True)

        x, y = next(iter(train_loader))
        betas = self.make_beta_schedule()

        x = x.view(16, 1, 28, 28)
        x_n = self.q_sample(betas, x)
        self.p_sample(betas, x_n)
        self.p_sample(betas)
