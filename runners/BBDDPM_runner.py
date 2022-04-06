import torch
import yaml
import os
import argparse

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid

from model.DDPM.ddpmnet import DDPMNet
from model.BrownianBridge.BrownianBridge import BrownianBridgeNet, CDiffuseBrownianBridgeNet, \
    CDiffuseBrownianBridgeNet_2
from runners.utils import get_optimizer, get_dataset, make_dirs, mkdir

from tqdm.autonotebook import tqdm
from PIL import Image


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


class BBDDPMRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config

        self.args.image_path, self.args.model_path, self.args.log_path, self.args.sample_path, \
        self.args.now = make_dirs(self.args, 'BBDDPM', 'test')
        self.grid_size = 4

    @torch.no_grad()
    def save_images(self, all_samples, sample_path, grid_size=4):
        # pdb.set_trace()
        imgs = []
        for i, sample in enumerate(tqdm(all_samples, total=len(all_samples), desc='saving images')):
            if i % 2 == 0 or i % 20 == 0:
                sample = sample.view(self.config.training.batch_size, self.config.data.channels, self.config.data.image_size,
                                     self.config.data.image_size)

                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                image_grid = make_grid(sample, nrow=self.grid_size)
                im = Image.fromarray(
                    image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                )
                if i % 2 == 0:
                    imgs.append(im)

                if i % 20 == 0:
                    im.save(os.path.join(sample_path, 'image_{}.png'.format(i)))

        image_grid = make_grid(all_samples[len(all_samples) - 1], nrow=self.grid_size)
        im = Image.fromarray(
            image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        )
        im.save(os.path.join(sample_path, 'image_{}.png'.format(len(all_samples))))

        imgs[0].save(os.path.join(sample_path, "movie.gif"), save_all=True, append_images=imgs[1:],
                     duration=1, loop=0)

    @torch.no_grad()
    def brownianbridge_sample(self, bbnet, sample_path, suffix, x_trans, x, type='backward', step=None):
        sample_path = os.path.join(sample_path, suffix)
        mkdir(sample_path)

        image_grid = make_grid(x_trans, nrow=self.grid_size)
        save_image(image_grid, os.path.join(sample_path, 'condition.png'))
        image_grid = make_grid(x, nrow=self.grid_size)
        save_image(image_grid, os.path.join(sample_path, 'ground_truth.png'))

        if type == 'backward':
            all_samples = bbnet.p_sample_loop(x_trans, x, step)
        else:
            all_samples = bbnet.q_sample_loop(x, x_trans)

        self.save_images(all_samples, sample_path)
        return all_samples[-1]

    @torch.no_grad()
    def ddpm_sample(self, ddpmnet, sample_path, suffix, x=None, step=None, grid_size=4):
        sample_path = os.path.join(sample_path, suffix)
        mkdir(sample_path)

        if x is not None:
            if step is None:
                step = ddpmnet.config.model.n_steps - 1
            t = torch.full((x.shape[0],), step).to(ddpmnet.config.device)

            perturbed_x = ddpmnet.q_sample(x, t)
            all_samples = ddpmnet.p_sample_loop(perturbed_x.shape, perturbed_x, step)
        else:
            all_samples = ddpmnet.sample(grid_size ** 2)
        self.save_images(all_samples, sample_path)

    @torch.no_grad()
    def test(self):
        with open(os.path.join("./configs", self.config.ddpm_config), 'r') as f:
            ddpm_config = yaml.load(f, Loader=yaml.FullLoader)
        ddpm_config = dict2namespace(ddpm_config)
        ddpm_config.device = torch.device('cuda:0')

        ddpmnet = DDPMNet(ddpm_config).to(ddpm_config.device)
        states = torch.load(os.path.join(self.config.ddpm_path))
        ddpmnet.load_state_dict(states[0])

        with open(os.path.join("./configs", self.config.bb_config), 'r') as f:
            bb_config = yaml.load(f, Loader=yaml.FullLoader)
        bb_config = dict2namespace(bb_config)
        bb_config.device = torch.device('cuda:0')

        brownian_bridge_net = None
        if bb_config.model.name == 'BB':
            brownian_bridge_net = BrownianBridgeNet(bb_config).to(bb_config.device)
        elif bb_config.model.name == 'CDBB':
            brownian_bridge_net = CDiffuseBrownianBridgeNet(bb_config).to(bb_config.device)
        elif bb_config.model.name == 'CDBB_2':
            brownian_bridge_net = CDiffuseBrownianBridgeNet_2(bb_config).to(bb_config.device)

        states = torch.load(os.path.join(self.config.bb_path))
        brownian_bridge_net.load_state_dict(states[0])

        train_dataset, test_dataset = get_dataset(self.config.data)
        train_loader = DataLoader(train_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                  num_workers=8, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 num_workers=8, drop_last=True)

        for i in range(1):
            sample_path = os.path.join(self.args.sample_path, str(i))
            mkdir(sample_path)

            train_sample_path = os.path.join(sample_path, 'train_sample')
            X, X_cond = next(iter(train_loader))
            X = X.to(self.config.device)
            X = X / 256. * 255. + torch.rand_like(X) / 256.

            X_cond = X_cond.to(self.config.device)
            X_cond = X_cond / 256. * 255. + torch.rand_like(X_cond) / 256.

            mid_sample = self.brownianbridge_sample(brownian_bridge_net, train_sample_path, 'Brownian_Bridge', X_cond, X)
            self.ddpm_sample(ddpmnet, train_sample_path, 'DDPM', mid_sample, step=300)

            test_sample_path = os.path.join(sample_path, 'test_sample')
            X, X_cond = next(iter(test_loader))
            X = X.to(self.config.device)
            X = X / 256. * 255. + torch.rand_like(X) / 256.

            X_cond = X_cond.to(self.config.device)
            X_cond = X_cond / 256. * 255. + torch.rand_like(X_cond) / 256.

            mid_sample = self.brownianbridge_sample(brownian_bridge_net, test_sample_path, 'Brownian_Bridge', X_cond,
                                                    X)
            self.ddpm_sample(ddpmnet, test_sample_path, 'DDPM', mid_sample, step=300)
