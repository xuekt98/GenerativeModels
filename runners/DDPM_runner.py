import traceback

import torch
import os

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid

from model.DDPM.ddpmnet import DDPMNet
from runners.utils import get_optimizer, get_dataset, make_dirs, mkdir

from tqdm.autonotebook import tqdm
from PIL import Image


class DDPMRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config

        self.args.image_path, self.args.model_path, self.args.log_path, self.args.sample_path, \
        self.args.now = make_dirs(self.args, 'DDPM', 'test')
        self.grid_size = 4

    @torch.no_grad()
    def save_images(self, all_samples, sample_path, grid_size=4):
        imgs = []
        for i, sample in enumerate(tqdm(all_samples, total=len(all_samples), desc='saving images')):
            if i % 5 == 0 or i % 50 == 0:
                sample = sample.view(self.config.training.batch_size, self.config.data.channels,
                                     self.config.data.image_size, self.config.data.image_size)
                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                image_grid = make_grid(sample, nrow=grid_size)
                im = Image.fromarray(
                    image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                )
                if i % 5 == 0:
                    imgs.append(im)

                if i % 50 == 0:
                    im.save(os.path.join(sample_path, 'image_{}.png'.format(i)))

        image_grid = make_grid(all_samples[len(all_samples) - 1], nrow=self.grid_size)
        im = Image.fromarray(
            image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        )
        im.save(os.path.join(sample_path, 'image_{}.png'.format(len(all_samples))))

        imgs[0].save(os.path.join(sample_path, "movie.gif"), save_all=True, append_images=imgs[1:],
                     duration=1, loop=0)

    @torch.no_grad()
    def ddpm_sample(self, ddpmnet, sample_path, suffix, x=None, step=None, grid_size=4):
        sample_path = os.path.join(sample_path, suffix)
        mkdir(sample_path)

        if x is not None:
            if step is None:
                step = self.config.model.n_steps - 1
            t = torch.full((x.shape[0],), step).to(self.config.device)

            perturbed_x = ddpmnet.q_sample(x, t)
            all_samples = ddpmnet.p_sample_loop(perturbed_x.shape, perturbed_x, step)
        else:
            all_samples = ddpmnet.sample(grid_size ** 2)
        self.save_images(all_samples, sample_path)

    def train(self):
        writer = SummaryWriter(self.args.log_path)

        train_dataset, test_dataset = get_dataset(self.config.data)
        train_loader = DataLoader(train_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                  num_workers=8, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 num_workers=8, drop_last=True)

        ddpmnet = DDPMNet(self.config).to(self.config.device)

        optimizer = get_optimizer(self.config.optimizer, ddpmnet.parameters())
        if self.args.load_model:
            states = torch.load(os.path.join(self.args.model_path, 'checkpoint.pth'))
            ddpmnet.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2000,
                                                               verbose=True, threshold=0.002, threshold_mode='rel',
                                                               cooldown=2000)

        step = self.args.load_iter
        pbar = tqdm(range(self.config.training.n_epochs), initial=0, dynamic_ncols=True, smoothing=0.01)
        for epoch in pbar:
            for i, (X, y) in enumerate(train_loader):
                try:
                    if step >= self.config.training.n_iters:
                        return 0
                    step += 1
                    ddpmnet.train()
                    X = X.to(self.config.device)
                    X = X / 256. * 255. + torch.rand_like(X) / 256.

                    loss = ddpmnet(X)

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

                    if step % 100 == 0:
                        ddpmnet.eval()
                        test_X, test_Y = next(iter(test_loader))
                        test_X = test_X.to(self.config.device)
                        test_X = test_X / 256. * 255. + torch.rand_like(test_X) / 256.

                        test_loss = ddpmnet(test_X)
                        writer.add_scalar('test_loss', test_loss, step)

                    if step % 1000 == 0:
                        ddpmnet.eval()
                        sample_path = os.path.join(self.args.sample_path, str(step))
                        mkdir(sample_path)
                        test_X, _ = next(iter(test_loader))
                        test_X = test_X.to(self.config.device)
                        self.ddpm_sample(ddpmnet, sample_path, 'train_sample', X)
                        self.ddpm_sample(ddpmnet, sample_path, 'test_sample', test_X)
                        self.ddpm_sample(ddpmnet, sample_path, 'random_sample')
                    # if step > 0 and step % 5000 == 0:
                    if step % 10000 == 0:
                        states = [
                            ddpmnet.state_dict(),
                            optimizer.state_dict(),
                        ]
                        torch.save(states, os.path.join(self.args.model_path, 'checkpoint_{}.pth'.format(step)))
                        torch.save(states, os.path.join(self.args.model_path, 'checkpoint.pth'))
                except BaseException as e:
                    print('Exception save model start!!!')
                    states = [
                        ddpmnet.state_dict(),
                        optimizer.state_dict(),
                    ]
                    torch.save(states, os.path.join(self.args.model_path, 'checkpoint_{}.pth'.format(step)))
                    torch.save(states, os.path.join(self.args.model_path, 'checkpoint.pth'))
                    print('Exception save model success!!!')
                    print('str(Exception):\t', str(Exception))
                    print('str(e):\t\t', str(e))
                    print('repr(e):\t', repr(e))
                    print('traceback.print_exc():')
                    traceback.print_exc()
                    print('traceback.format_exc():\n%s' % traceback.format_exc())
        states = [
            ddpmnet.state_dict(),
            optimizer.state_dict(),
        ]
        torch.save(states, os.path.join(self.args.model_path, 'checkpoint_{}.pth'.format(step)))
        torch.save(states, os.path.join(self.args.model_path, 'checkpoint.pth'))