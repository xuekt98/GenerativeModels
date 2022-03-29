import numpy as np
import os
import torch
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from losses.dsm import anneal_dsm_score_estimation
from model.NCSN.langevindynamics import anneal_Langevin_dynamics
from model.NCSN.scorenet import CondRefineNetDilated
from runners.utils import get_optimizer, get_dataset, make_dirs, mkdir
from PIL import Image


class NCSNRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config

        self.args.image_path, self.args.model_path, self.args.log_path, self.args.sample_path,\
        self.args.now = make_dirs(self.args, 'NCSN')

    def logit_transform(self, image, lam=1e-6):
        image = lam + (1 - 2 * lam) * image
        return torch.log(image) - torch.log1p(-image)

    def anneal_langevin_sample(self, scorenet, sigmas, step=None):
        sample_path = self.args.sample_path
        if step is not None:
            sample_path = os.path.join(sample_path, str(step))
        mkdir(sample_path)

        grid_size = 5
        samples = torch.rand(grid_size ** 2, 1, 28, 28, device=self.config.device)
        all_samples = anneal_Langevin_dynamics(samples, scorenet, sigmas, 100, 0.00002)

        imgs = []
        for i, sample in enumerate(tqdm(all_samples, total=len(all_samples), desc='saving images')):
            sample = sample.view(grid_size ** 2, self.config.data.channels, self.config.data.image_size,
                                 self.config.data.image_size)
            if self.config.data.logit_transform:
                sample = torch.sigmoid(sample)

            image_grid = make_grid(sample, nrow=grid_size)
            if i % 5 == 0:
                im = Image.fromarray(
                    image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                imgs.append(im)

            if i % 50 == 0:
                save_image(image_grid, os.path.join(sample_path, 'image_{}.png'.format(i)))
            # torch.save(sample, os.path.join(self.args.image_folder, 'image_raw_{}.pth'.format(i)))
        imgs[0].save(os.path.join(sample_path, "movie.gif"), save_all=True, append_images=imgs[1:],
                     duration=1, loop=0)

    def train(self):
        writer = SummaryWriter(self.args.log_path)

        train_dataset, test_dataset = get_dataset(self.config.data)
        train_loader = DataLoader(train_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                  num_workers=8, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 num_workers=8, drop_last=True)

        # scorenet = SimpleScoreNet(self.config).to(self.config.device)
        scorenet = CondRefineNetDilated(self.config).to(self.config.device)

        optimizer = get_optimizer(self.config.optimizer, scorenet.parameters())
        if self.args.load_model:
            states = torch.load(os.path.join(self.args.model_path, 'checkpoint.pth'))
            scorenet.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2000,
                                                               verbose=True, threshold=0.002, threshold_mode='rel',
                                                               cooldown=2000)

        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                               self.config.model.num_classes))).float().to(self.config.device)

        step = self.args.load_iter
        pbar = tqdm(range(self.config.training.n_epochs), initial=0, dynamic_ncols=True, smoothing=0.01)
        for epoch in pbar:
            for i, (X, y) in enumerate(train_loader):
                try:
                    if step >= self.config.training.n_iters:
                        return 0
                    step += 1
                    scorenet.train()
                    X = X.to(self.config.device)
                    X = X / 256. * 255. + torch.rand_like(X) / 256.
                    if self.config.data.logit_transform:
                        X = self.logit_transform(X)

                    labels = torch.randint(0, len(sigmas), (X.shape[0],), device=X.device)
                    loss = anneal_dsm_score_estimation(scorenet, X, labels, sigmas, self.config.training.anneal_power)

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
                        scorenet.eval()
                        test_X, test_Y = next(iter(test_loader))
                        test_X = test_X.to(self.config.device)
                        test_X = test_X / 256. * 255. + torch.rand_like(test_X) / 256.

                        test_labels = torch.randint(0, len(sigmas), (test_X.shape[0],), device=test_X.device)
                        test_loss = anneal_dsm_score_estimation(scorenet, X, test_labels, sigmas,
                                                                self.config.training.anneal_power)
                        writer.add_scalar('test_loss', test_loss, step)

                    if step % 500 == 0:
                        self.anneal_langevin_sample(scorenet, sigmas, step)
                    # if step > 0 and step % 5000 == 0:
                    if step % 10000 == 0:
                        states = [
                            scorenet.state_dict(),
                            optimizer.state_dict(),
                        ]
                        torch.save(states, os.path.join(self.args.model_path, 'checkpoint_{}.pth'.format(step)))
                        torch.save(states, os.path.join(self.args.model_path, 'checkpoint.pth'))
                except BaseException as e:
                    print('Exception save model start!!!')
                    states = [
                        scorenet.state_dict(),
                        optimizer.state_dict(),
                    ]
                    torch.save(states, os.path.join(self.args.model_path, 'checkpoint_{}.pth'.format(step)))
                    torch.save(states, os.path.join(self.args.model_path, 'checkpoint.pth'))
                    print('Exception save model success!!!')
                    print(e)


