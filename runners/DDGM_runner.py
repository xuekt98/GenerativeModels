import os
import torch
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from model.DDGM.ddgmnet import DDGMNet
from runners.utils import get_optimizer, get_dataset, make_dirs, mkdir
from PIL import Image


class DDGMRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config

        self.args.image_path, self.args.model_path, self.args.log_path, self.args.sample_path,\
        self.args.now = make_dirs(self.args, 'DDGM')

    def ddgm_sample(self, ddgmnet, samples=None, step=None):
        sample_path = self.args.sample_path
        if step is not None:
            sample_path = os.path.join(sample_path, str(step))
        mkdir(sample_path)
        sample_path = os.path.join(sample_path, 'random_sample')
        mkdir(sample_path)

        # pdb.set_trace()
        grid_size = 4
        if samples is None:
            samples = torch.rand(grid_size ** 2, 1, 28, 28, device=self.config.device)

        all_samples = ddgmnet.sample(samples)
        self.save_images(all_samples, sample_path)

    def save_images(self, all_samples, sample_path, grid_size=4):
        imgs = []
        for i, sample in enumerate(tqdm(all_samples, total=len(all_samples), desc='saving images')):
            sample = sample.view(grid_size**2, self.config.data.channels, self.config.data.image_size,
                                 self.config.data.image_size)
            if self.config.data.logit_transform:
                sample = torch.sigmoid(sample)

            image_grid = make_grid(sample, nrow=grid_size)
            if i % 2 == 0:
                im = Image.fromarray(
                    image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                imgs.append(im)

            if i % 20 == 0:
                save_image(image_grid, os.path.join(sample_path, 'image_{}.png'.format(i)))

        imgs[0].save(os.path.join(sample_path, "movie.gif"), save_all=True, append_images=imgs[1:],
                     duration=1, loop=0)

    def train(self):
        writer = SummaryWriter(self.args.log_path)

        train_dataset, test_dataset = get_dataset(self.config.data)
        train_loader = DataLoader(train_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                  num_workers=8, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 num_workers=8, drop_last=True)

        ddgmnet = DDGMNet(self.config).to(self.config.device)

        optimizer = get_optimizer(self.config.optimizer, ddgmnet.parameters())
        if self.args.load_model:
            states = torch.load(os.path.join(self.args.model_path, 'checkpoint.pth'))
            ddgmnet.load_state_dict(states[0])
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
                    ddgmnet.train()
                    X = X.to(self.config.device)
                    X = X / 256. * 255. + torch.rand_like(X) / 256.

                    loss, zs, mus = ddgmnet(X)

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
                        ddgmnet.eval()
                        test_X, test_Y = next(iter(test_loader))
                        test_X = test_X.to(self.config.device)
                        test_X = test_X / 256. * 255. + torch.rand_like(test_X) / 256.

                        test_loss, _, _ = ddgmnet(test_X)
                        writer.add_scalar('test_loss', test_loss, step)

                    if step % 500 == 0:
                        image_path = os.path.join(self.args.sample_path, str(step))
                        mkdir(image_path)

                        forward_path = os.path.join(image_path, 'forward')
                        mkdir(forward_path)
                        backward_path = os.path.join(image_path, 'backward')
                        mkdir(backward_path)
                        perturbed_path = os.path.join(image_path, 'perturbed_sample')
                        mkdir(perturbed_path)

                        mus.reverse()
                        self.save_images(zs, forward_path)
                        self.save_images(mus, backward_path)

                        # sample from froward perturbed noise
                        perturbed_samples = ddgmnet.sample(zs[-1])
                        self.save_images(perturbed_samples, perturbed_path)

                        # sample from random noise
                        self.ddgm_sample(ddgmnet, step=step)
                    # if step > 0 and step % 5000 == 0:
                    if step % 10000 == 0:
                        states = [
                            ddgmnet.state_dict(),
                            optimizer.state_dict(),
                        ]
                        torch.save(states, os.path.join(self.args.model_path, 'checkpoint_{}.pth'.format(step)))
                        torch.save(states, os.path.join(self.args.model_path, 'checkpoint.pth'))
                except BaseException as e:
                    print('Exception save model start!!!')
                    states = [
                        ddgmnet.state_dict(),
                        optimizer.state_dict(),
                    ]
                    torch.save(states, os.path.join(self.args.model_path, 'checkpoint_{}.pth'.format(step)))
                    torch.save(states, os.path.join(self.args.model_path, 'checkpoint.pth'))
                    print('Exception save model success!!!')
                    print(e)


