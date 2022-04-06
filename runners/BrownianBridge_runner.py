import pdb
import traceback

import torch
import os

import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid

from model.BrownianBridge.BrownianBridge import \
    BrownianBridgeNet, \
    BrownianBridgeNet_2,\
    CDiffuseBrownianBridgeNet, \
    CDiffuseBrownianBridgeNet_2
from runners.utils import get_optimizer, get_dataset, make_dirs, mkdir

from tqdm.autonotebook import tqdm
from PIL import Image


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class BrownianBridgeRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config

        self.args.image_path, self.args.model_path, self.args.log_path, self.args.sample_path, \
        self.args.now = make_dirs(self.args, 'BrownianBridge', 'test')
        self.grid_size = 4

    def save_images(self, all_samples, sample_path, grid_size=4):
        # pdb.set_trace()
        imgs = []
        for i, sample in enumerate(tqdm(all_samples, total=len(all_samples), desc='saving images')):
            if i % 5 == 0 or i % 50 == 0:
                sample = sample.view(self.config.training.batch_size, self.config.data.channels, self.config.data.image_size,
                                     self.config.data.image_size)

                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                image_grid = make_grid(sample, nrow=self.grid_size)
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

    def brownianbridge_sample(self, bbnet, sample_path, suffix, x_trans, x, type='backward', step=None):
        sample_path = os.path.join(sample_path, suffix)
        mkdir(sample_path)

        image_grid = make_grid(x_trans, nrow=self.grid_size)
        save_image(image_grid, os.path.join(sample_path, 'condition.png'))
        image_grid = make_grid(x, nrow=4)
        save_image(image_grid, os.path.join(sample_path, 'ground_truth.png'))

        if type == 'backward':
            all_samples = bbnet.p_sample_loop(x_trans, x, step)
        else:
            all_samples = bbnet.q_sample_loop(x, x_trans)

        self.save_images(all_samples, sample_path)

    def train(self):
        writer = SummaryWriter(self.args.log_path)

        train_dataset, test_dataset = get_dataset(self.config.data)
        train_loader = DataLoader(train_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                  num_workers=8, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 num_workers=8, drop_last=True)
        if self.config.model.name == 'BB':
            brownian_bridge_net = BrownianBridgeNet(self.config).to(self.config.device)
        if self.config.model.name == 'BB_2':
            brownian_bridge_net = BrownianBridgeNet_2(self.config).to(self.config.device)
        elif self.config.model.name == 'CDBB':
            brownian_bridge_net = CDiffuseBrownianBridgeNet(self.config).to(self.config.device)
        elif self.config.model.name == 'CDBB_2':
            brownian_bridge_net = CDiffuseBrownianBridgeNet_2(self.config).to(self.config.device)

        optimizer = get_optimizer(self.config.optimizer, brownian_bridge_net.parameters())
        if self.args.load_model:
            states = torch.load(os.path.join(self.args.model_path, 'checkpoint.pth'))
            brownian_bridge_net.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=500,
                                                               verbose=True, threshold=0.005, threshold_mode='rel',
                                                               cooldown=500, min_lr=5e-8)

        # cond_transform = torchvision.transforms.RandomHorizontalFlip(p=1.0)
        # gaussian_blur_transform = torchvision.transforms.GaussianBlur(7, sigma=(100, 100))

        step = self.args.load_iter
        pbar = tqdm(range(self.config.training.n_epochs), initial=0, dynamic_ncols=True, smoothing=0.01)
        for epoch in pbar:
            for i, (X, X_cond) in enumerate(train_loader):
                try:
                    if step >= self.config.training.n_iters:
                        return 0
                    step += 1
                    brownian_bridge_net.train()
                    X = X.to(self.config.device)
                    # X = X / 256. * 255. + torch.rand_like(X) / 256.

                    X_cond = X_cond.to(self.config.device)
                    # X_cond = X_cond / 256. * 255. + torch.rand_like(X_cond) / 256.

                    loss = brownian_bridge_net(X, X_cond)
                    # loss = brownian_bridge_net(X_cond, X_cond)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step(loss)
                    pbar.set_description(
                        (
                            f'iter: {step} loss: {loss:.4f}'
                        )
                    )

                    writer.add_scalar('loss/train', loss, step)

                    if step % 100 == 0:
                        brownian_bridge_net.eval()
                        test_X, test_X_cond = next(iter(test_loader))
                        test_X = test_X.to(self.config.device)
                        # test_X = test_X / 256. * 255. + torch.rand_like(test_X) / 256.

                        test_X_cond = test_X_cond.to(self.config.device)
                        # test_X_cond = test_X_cond / 256. * 255. + torch.rand_like(test_X_cond) / 256.

                        # test_X_trans = cond_transform(test_X)

                        test_loss = brownian_bridge_net(test_X, test_X_cond)
                        # test_loss = brownian_bridge_net(test_X_cond, test_X_cond)
                        writer.add_scalar('loss/test', test_loss, step)

                    if step % 1000 == 0:
                        brownian_bridge_net.eval()
                        sample_path = os.path.join(self.args.sample_path, str(step))
                        mkdir(sample_path)
                        test_X, test_X_cond = next(iter(test_loader))
                        test_X = test_X.to(self.config.device)
                        # test_X = test_X / 256. * 255. + torch.rand_like(test_X) / 256.

                        test_X_cond = test_X_cond.to(self.config.device)
                        # test_X_cond = test_X_cond / 256. * 255. + torch.rand_like(test_X_cond) / 256.

                        #test_X_trans = cond_transform(test_X)

                        # self.brownianbridge_sample(brownian_bridge_net, sample_path, 'train_forward_sample', X_cond, X, type='forward')
                        self.brownianbridge_sample(brownian_bridge_net, sample_path, 'train_sample', X_cond, X)
                        # self.brownianbridge_sample(brownian_bridge_net, sample_path, 'test_forward_sample', test_X_cond, test_X, type='forward')
                        self.brownianbridge_sample(brownian_bridge_net, sample_path, 'test_sample', test_X_cond, test_X)

                        # self.brownianbridge_sample(brownian_bridge_net, sample_path, 'train_sample', X_cond, X_cond)
                        # self.brownianbridge_sample(brownian_bridge_net, sample_path, 'test_sample', test_X_cond, test_X_cond)
                    if step % 5000 == 0:
                        states = [
                            brownian_bridge_net.state_dict(),
                            optimizer.state_dict(),
                        ]
                        torch.save(states, os.path.join(self.args.model_path, 'checkpoint_{}.pth'.format(step)))
                        torch.save(states, os.path.join(self.args.model_path, 'checkpoint.pth'))
                except BaseException as e:
                    print('Exception save model start!!!')
                    states = [
                        brownian_bridge_net.state_dict(),
                        optimizer.state_dict(),
                    ]
                    torch.save(states, os.path.join(self.args.model_path, 'checkpoint_exp.pth'))
                    print('Exception save model success!!!')
                    print('str(Exception):\t', str(Exception))
                    print('str(e):\t\t', str(e))
                    print('repr(e):\t', repr(e))
                    print('traceback.print_exc():')
                    traceback.print_exc()
                    print('traceback.format_exc():\n%s' % traceback.format_exc())

        states = [
            brownian_bridge_net.state_dict(),
            optimizer.state_dict(),
        ]
        torch.save(states, os.path.join(self.args.model_path, 'checkpoint_{}.pth'.format(step)))
        torch.save(states, os.path.join(self.args.model_path, 'checkpoint.pth'))

    @torch.no_grad()
    def test(self):
        train_dataset, test_dataset = get_dataset(self.config.data)
        train_loader = DataLoader(train_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                  num_workers=8, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 num_workers=8, drop_last=True)

        brownian_bridge_net = BrownianBridgeNet(self.config).to(self.config.device)

        optimizer = get_optimizer(self.config.optimizer, brownian_bridge_net.parameters())
        if self.args.load_model:
            states = torch.load(os.path.join(self.args.model_path, 'checkpoint.pth'))
            brownian_bridge_net.load_state_dict(states[0])

        brownian_bridge_net.eval()
        for i in tqdm(range(1), initial=0, dynamic_ncols=True, smoothing=0.01):
            sample_path = os.path.join(self.args.image_path, str(i))

            X, X_cond = next(iter(train_loader))
            X = X.to(self.config.device)
            # X = X / 256. * 255. + torch.rand_like(X) / 256.

            X_cond = X_cond.to(self.config.device)
            # X_cond = X_cond / 256. * 255. + torch.rand_like(X_cond) / 256.
            self.brownianbridge_sample(brownian_bridge_net, sample_path, 'train_sample', X_cond, X, step=100)

            test_X, test_X_cond = next(iter(test_loader))
            test_X = test_X.to(self.config.device)
            # test_X = test_X / 256. * 255. + torch.rand_like(test_X) / 256.

            test_X_cond = test_X_cond.to(self.config.device)
            # test_X_cond = test_X_cond / 256. * 255. + torch.rand_like(test_X_cond) / 256.
            self.brownianbridge_sample(brownian_bridge_net, sample_path, 'test_sample', test_X_cond, test_X, step=100)