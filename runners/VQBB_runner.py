import traceback
import torch
import os
import pdb
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid

from model.BrownianBridge.BrownianBridge import BrownianBridgeNet, BrownianBridgeNet_2, CDiffuseBrownianBridgeNet, \
    CDiffuseBrownianBridgeNet_2
from model.VQBrownianBridge.discriminator import NLayerDiscriminator, weights_init
from model.VQBrownianBridge.lpips import LPIPS
from model.VQBrownianBridge.vqmodel import VQModel
from runners.utils import get_optimizer, get_dataset, make_dirs, mkdir

from tqdm.autonotebook import tqdm
from PIL import Image


def get_vqmodel_config_dict(vqmodel_config):
    dict = {
        # 'ckpt_path': vqmodel_config.ckpt_path,
        'embed_dim': vqmodel_config.embed_dim,
        'n_embed': vqmodel_config.n_embed,
        'ed_config': {
            'z_channels': vqmodel_config.edconfig.z_channels,
            'resolution': vqmodel_config.edconfig.resolution,
            'in_channels': vqmodel_config.edconfig.in_channels,
            'out_channels': vqmodel_config.edconfig.out_channels,
            'map_channels': vqmodel_config.edconfig.map_channels,
            'ch_mult': vqmodel_config.edconfig.ch_mult,
            'num_res_blocks': vqmodel_config.edconfig.num_res_blocks,
            'attn_resolutions': vqmodel_config.edconfig.attn_resolutions,
            'dropout': vqmodel_config.edconfig.dropout
        }
    }
    return dict

def get_discriminator_config_dict(discriminator_config):
    dict = {
        'input_nc': discriminator_config.input_nc,
        'ndf': discriminator_config.ndf,
        'n_layers': discriminator_config.n_layers,
        'use_actnorm': discriminator_config.use_actnorm
    }
    return dict

class VQBBRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config

        self.args.image_path, self.args.model_path, self.args.log_path, self.args.sample_path, \
        self.args.now = make_dirs(self.args, 'VQBB', 'bdd')
        self.grid_size = 4
        self.writer = SummaryWriter(self.args.log_path)

    def save_image(self, sample, sample_path, image_name, grid_size=2):
        mkdir(sample_path)
        image_grid = make_grid(sample, nrow=grid_size)
        im = Image.fromarray(
            image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        )
        im.save(os.path.join(sample_path, image_name))

    def train_vqmodel(self, type='ori'):
        sample_path = os.path.join(self.args.sample_path, f'vqmodel_{type}')
        mkdir(sample_path)

        model_config = self.config.vqmodel

        train_dataset, test_dataset = get_dataset(model_config.data)
        train_loader = DataLoader(train_dataset, batch_size=model_config.training.batch_size, shuffle=True,
                                  num_workers=8, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=model_config.training.batch_size, shuffle=True,
                                 num_workers=8, drop_last=True)

        vqmodel_config = get_vqmodel_config_dict(model_config.model)
        vqmodel = VQModel(**vqmodel_config).to(self.config.device)

        optimizer = get_optimizer(model_config.optimizer, vqmodel.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1000,
                                                               verbose=True, threshold=0.005, threshold_mode='rel',
                                                               cooldown=1000, min_lr=1e-7)

        if model_config.load_vqmodel:
            states = torch.load(os.path.join(self.args.model_path, f'vqmodel_{type}.pth'))
            vqmodel.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])
            scheduler.load_state_dict(states[2])

        step = self.args.load_iter
        pbar = tqdm(range(model_config.training.n_epochs), initial=0, dynamic_ncols=True, smoothing=0.01)

        for epoch in pbar:
            for i, (X, X_cond) in enumerate(train_loader):
                if step > model_config.training.n_iters:
                    break
                step += 1
                try:
                    if type == 'ori':
                        image = X.to(self.config.device)
                    else:
                        image = X_cond.to(self.config.device)

                    vqmodel.train()
                    optimizer.zero_grad()

                    image_rec, qloss = vqmodel(image)
                    recloss = F.mse_loss(image_rec, image)
                    # recloss = (image_rec.contiguous() - image.contiguous()).abs().mean()
                    loss = recloss + qloss.mean()

                    loss.backward()
                    optimizer.step()
                    scheduler.step(loss)

                    pbar.set_description(
                        (
                            f'iter: {step} loss: {loss:.4f} recloss: {recloss:.4f} qloss: {qloss.mean():.4f}'
                        )
                    )
                    self.writer.add_scalar(f'vqmodel_{type}/train', loss, step)

                    test_image, test_image_rec = None, None
                    if step % 100 == 0:
                        vqmodel.eval()
                        test_X, test_X_cond = next(iter(test_loader))
                        if type == 'ori':
                            test_image = test_X.to(self.config.device)
                        else:
                            test_image = test_X_cond.to(self.config.device)

                        test_image_rec, test_qloss = vqmodel(test_image)
                        test_recloss = torch.abs(test_image_rec.contiguous() - test_image.contiguous())
                        test_loss = test_recloss.mean() + test_qloss.mean()
                        self.writer.add_scalar(f'vqmodel_{type}/test', test_loss, step)

                    if step % 1000 == 0:
                        image_path = os.path.join(sample_path, str(step))
                        mkdir(image_path)

                        self.save_image(image, os.path.join(image_path, 'train'), 'img_ori.png')
                        self.save_image(image_rec, os.path.join(image_path, 'train'), 'img_rec.png')
                        self.save_image(test_image, os.path.join(image_path, 'test'), 'img_ori.png')
                        self.save_image(test_image_rec, os.path.join(image_path, 'test'), 'img_rec.png')

                    if step % 5000 == 0:
                        states = [
                            vqmodel.state_dict(),
                            optimizer.state_dict(),
                            scheduler.state_dict()
                        ]
                        torch.save(states, os.path.join(self.args.model_path, f'vqmodel_{type}_{step}.pth'))
                except BaseException as e:
                    print('Exception save model start!!!')
                    states = [
                        vqmodel.state_dict(),
                        optimizer.state_dict(),
                        scheduler.state_dict()
                    ]
                    torch.save(states, os.path.join(self.args.model_path, f'vqmodel_{type}_exp.pth'))
                    print('Exception save model success!!!')
                    print('str(Exception):\t', str(Exception))
                    print('str(e):\t\t', str(e))
                    print('repr(e):\t', repr(e))
                    print('traceback.print_exc():')
                    traceback.print_exc()
                    print('traceback.format_exc():\n%s' % traceback.format_exc())
        states = [
            vqmodel.state_dict(),
            optimizer.state_dict(),
            scheduler.state_dict()
        ]
        torch.save(states, os.path.join(self.args.model_path, f'vqmodel_{type}_{step}.pth'))

    def train_vqgan(self, type='ori'):
        sample_path = os.path.join(self.args.sample_path, f'vqmodel_{type}')
        mkdir(sample_path)

        model_config = self.config.vqmodel

        train_dataset, test_dataset = get_dataset(self.config.data)
        train_loader = DataLoader(train_dataset, batch_size=model_config.training.batch_size, shuffle=True,
                                  num_workers=8, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=model_config.training.batch_size, shuffle=True,
                                 num_workers=8, drop_last=True)

        vqmodel_config = get_vqmodel_config_dict(model_config.model)
        vqmodel = VQModel(**vqmodel_config).to(self.config.device)

        optimizer = get_optimizer(model_config.g_optimizer, vqmodel.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1000,
                                                               verbose=True, threshold=0.005, threshold_mode='rel',
                                                               cooldown=1000, min_lr=5e-8)

        discriminator_config = get_discriminator_config_dict(model_config.discriminator)
        discriminator = NLayerDiscriminator(**discriminator_config).to(self.config.device)
        discriminator.apply(weights_init)

        d_optimizer = get_optimizer(model_config.d_optimizer, discriminator.parameters())
        d_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(d_optimizer, mode='min', factor=0.5, patience=1000,
                                                               verbose=True, threshold=0.005, threshold_mode='rel',
                                                               cooldown=1000, min_lr=5e-8)


        if model_config.load_vqmodel:
            states = torch.load(os.path.join(self.args.model_path, f'vqmodel_{type}.pth'))
            vqmodel.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])
            scheduler.load_state_dict(states[2])
            discriminator.load_state_dict(states[3])
            d_optimizer.load_state_dict(states[4])
            d_scheduler.load_state_dict(states[5])

        perceptual_loss = LPIPS().to(self.config.device).eval()

        step = self.args.load_iter
        pbar = tqdm(range(model_config.training.n_epochs), initial=0, dynamic_ncols=True, smoothing=0.01)

        def adopt_weight(weight, global_step, threshold=0, value=0.):
            if global_step < threshold:
                weight = value
            return weight

        def hinge_d_loss(logits_real, logits_fake):
            loss_real = torch.mean(F.relu(1. - logits_real))
            loss_fake = torch.mean(F.relu(1. + logits_fake))
            d_loss = 0.5 * (loss_real + loss_fake)
            return d_loss

        def calculate_adaptive_weight(nll_loss, g_loss, last_layer):
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

            d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
            d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
            d_weight = d_weight * 0.3
            return d_weight

        threshold = 200000
        for epoch in pbar:
            for i, (X, X_cond) in enumerate(train_loader):
                if step > model_config.training.n_iters:
                    break
                step += 1
                try:
                    if type == 'ori':
                        image = X.to(self.config.device)
                    else:
                        image = X_cond.to(self.config.device)

                    vqmodel.train()
                    # optimize dircriminator
                    image_rec, q_loss = vqmodel(image)
                    rec_loss = (image.contiguous() - image_rec.contiguous()).abs()
                    p_loss = perceptual_loss(image.contiguous(), image_rec.contiguous())
                    nll_loss = rec_loss + p_loss
                    nll_loss = torch.mean(nll_loss)
                    disc_factor = adopt_weight(1.0, step, threshold=threshold)
                    logits_fake = discriminator(image_rec.contiguous())
                    g_loss = -torch.mean(logits_fake)
                    dd_weight = calculate_adaptive_weight(nll_loss, g_loss, vqmodel.decoder.conv_out.weight)
                    loss = nll_loss + dd_weight * disc_factor * g_loss + q_loss.mean()
                    # loss = nll_loss + disc_factor * g_loss + q_loss.mean()
                    # loss = nll_loss + qloss.mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # if step > threshold:
                    #     scheduler.step(loss)

                    d_loss = 0.
                    if step > threshold:
                        discriminator.train()
                        logits_real = discriminator(image.contiguous().detach())
                        logits_fake = discriminator(image_rec.contiguous().detach())
                        d_loss = disc_factor * hinge_d_loss(logits_real, logits_fake)
                        d_optimizer.zero_grad()
                        d_loss.backward()
                        d_optimizer.step()
                        # d_scheduler.step(d_loss)
                        self.writer.add_scalar(f'vqmodel_{type}/train/d/d_loss', d_loss, step)
                        self.writer.add_scalar(f'vqmodel_{type}/train/d/logits_real', logits_real.detach().mean(), step)
                        self.writer.add_scalar(f'vqmodel_{type}/train/d/logits_fake', logits_fake.detach().mean(), step)

                    pbar.set_description(
                        (
                            f'iter: {step} loss: {loss:.4f} dloss: {d_loss:.4f} g_loss: {g_loss:.4f}'
                        )
                    )
                    self.writer.add_scalar(f'vqmodel_{type}/train/g/total_loss', loss, step)
                    self.writer.add_scalar(f'vqmodel_{type}/train/g/rec_loss', rec_loss.detach().mean(), step)
                    self.writer.add_scalar(f'vqmodel_{type}/train/g/q_loss', q_loss.detach().mean(), step)
                    self.writer.add_scalar(f'vqmodel_{type}/train/g/p_loss', p_loss.detach().mean(), step)
                    self.writer.add_scalar(f'vqmodel_{type}/train/g/g_loss', g_loss.detach(), step)
                    self.writer.add_scalar(f'vqmodel_{type}/train/g/nll_loss', nll_loss.detach().mean(), step)
                    self.writer.add_scalar(f'vqmodel_{type}/train/g/dd_weight', dd_weight.detach(), step)

                    with torch.no_grad():
                        test_image, test_image_rec = None, None
                        if step % 100 == 0:
                            vqmodel.eval()
                            test_X, test_X_cond = next(iter(test_loader))
                            if type == 'ori':
                                test_image = test_X.to(self.config.device)
                            else:
                                test_image = test_X_cond.to(self.config.device)

                            test_image_rec, test_qloss = vqmodel(test_image)
                            test_recloss = torch.abs(test_image_rec.contiguous() - test_image.contiguous())
                            test_loss = test_recloss.mean() + test_qloss.mean()
                            self.writer.add_scalar(f'vqmodel_{type}/test/loss', test_loss, step)
                            self.writer.add_scalar(f'vqmodel_{type}/test/recloss', test_recloss.mean(), step)
                            self.writer.add_scalar(f'vqmodel_{type}/test/qloss', test_qloss.mean(), step)

                        if step % 1000 == 0:
                        # if (step <= 5000 and step % 200 == 0) \
                        #         or (5000 < step <= 10000 and step % 1000 == 0) \
                        #         or (10000 < step <= 30000 and step % 2000 == 0) \
                        #         or (30000 < step <= 50000 and step % 5000 == 0) \
                        #         or (50000 < step and step % 10000 == 0):
                            # image_path = os.path.join(sample_path, str(step))
                            image_path = sample_path
                            mkdir(image_path)

                            self.save_image(image, os.path.join(image_path, 'train'), f'img_ori_{step}.png')
                            self.save_image(image_rec, os.path.join(image_path, 'train'), f'img_rec_{step}.png')
                            self.save_image(test_image, os.path.join(image_path, 'test'), f'img_ori_{step}.png')
                            self.save_image(test_image_rec, os.path.join(image_path, 'test'), f'img_rec_{step}.png')

                        if step % 5000 == 0:
                            states = [
                                vqmodel.state_dict(),
                                optimizer.state_dict(),
                                scheduler.state_dict(),
                                discriminator.state_dict(),
                                d_optimizer.state_dict(),
                                d_scheduler.state_dict()
                            ]
                            torch.save(states, os.path.join(self.args.model_path, f'vqmodel_{type}_{step}.pth'))
                            # torch.save(states, os.path.join(self.args.model_path, f'vqmodel_{type}.pth'))
                except BaseException as e:
                    print('Exception save model start!!!')
                    states = [
                        vqmodel.state_dict(),
                        optimizer.state_dict(),
                        scheduler.state_dict(),
                        discriminator.state_dict(),
                        d_optimizer.state_dict(),
                        d_scheduler.state_dict()
                    ]
                    torch.save(states, os.path.join(self.args.model_path, f'vqmodel_{type}_exp.pth'))
                    # torch.save(states, os.path.join(self.args.model_path, f'vqmodel_{type}.pth'))
                    print('Exception save model success!!!')
                    print('str(Exception):\t', str(Exception))
                    print('str(e):\t\t', str(e))
                    print('repr(e):\t', repr(e))
                    print('traceback.print_exc():')
                    traceback.print_exc()
                    print('traceback.format_exc():\n%s' % traceback.format_exc())
        states = [
            vqmodel.state_dict(),
            optimizer.state_dict(),
            scheduler.state_dict(),
            discriminator.state_dict(),
            d_optimizer.state_dict(),
            d_scheduler.state_dict()
        ]
        torch.save(states, os.path.join(self.args.model_path, f'vqmodel_{type}_{step}.pth'))
        # torch.save(states, os.path.join(self.args.model_path, f'vqmodel_{type}.pth'))

    def train_VQBB(self):
        sample_path = os.path.join(self.args.sample_path, f'vqbb')
        mkdir(sample_path)

        model_config = self.config.vqbbmodel

        train_dataset, test_dataset = get_dataset(self.config.data)
        train_loader = DataLoader(train_dataset, batch_size=model_config.training.batch_size, shuffle=True,
                                  num_workers=8, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=model_config.training.batch_size, shuffle=True,
                                 num_workers=8, drop_last=True)

        vqmodel_config = get_vqmodel_config_dict(self.config.vqmodel.model)
        vqmodel_ori = VQModel(**vqmodel_config).to(self.config.device)
        states = torch.load(os.path.join(self.args.model_path, f'vqmodel_ori.pth'))
        vqmodel_ori.load_state_dict(states[0])
        vqmodel_ori.requires_grad_(False)

        vqmodel_cond = VQModel(**vqmodel_config).to(self.config.device)
        states = torch.load(os.path.join(self.args.model_path, f'vqmodel_cond.pth'))
        vqmodel_cond.load_state_dict(states[0])
        vqmodel_cond.requires_grad_(False)

        if model_config.model.name == 'BB':
            brownian_bridge_net = BrownianBridgeNet(model_config).to(self.config.device)
        if model_config.model.name == 'BB_2':
            brownian_bridge_net = BrownianBridgeNet_2(model_config).to(self.config.device)
        elif model_config.model.name == 'CDBB':
            brownian_bridge_net = CDiffuseBrownianBridgeNet(model_config).to(self.config.device)
        elif model_config.model.name == 'CDBB_2':
            brownian_bridge_net = CDiffuseBrownianBridgeNet_2(model_config).to(self.config.device)

        optimizer = get_optimizer(model_config.optimizer, brownian_bridge_net.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1500,
                                                               verbose=True, threshold=0.005, threshold_mode='rel',
                                                               cooldown=1500, min_lr=5e-8)

        if model_config.load_vqbb:
            states = torch.load(os.path.join(self.args.model_path, f'vqbb.pth'))
            brownian_bridge_net.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])
            scheduler.load_state_dict(states[2])

        step = self.args.load_iter
        pbar = tqdm(range(model_config.training.n_epochs), initial=0, dynamic_ncols=True, smoothing=0.01)

        for epoch in pbar:
            for i, (X, X_cond) in enumerate(train_loader):
                if step > model_config.training.n_iters:
                    break
                step += 1
                try:
                    X = X.to(self.config.device)
                    X_cond = X_cond.to(self.config.device)

                    brownian_bridge_net.train()
                    optimizer.zero_grad()

                    X_quant, _, _ = vqmodel_ori.encode(X)
                    X_cond_quant, _, _ = vqmodel_cond.encode(X_cond)
                    # pdb.set_trace()
                    loss = brownian_bridge_net(X_quant, X_cond_quant)

                    loss.backward()
                    optimizer.step()
                    scheduler.step(loss)

                    pbar.set_description(
                        (
                            f'iter: {step} loss: {loss:.4f}'
                        )
                    )
                    self.writer.add_scalar(f'vqbb/train', loss, step)

                    test_image, test_image_rec = None, None
                    if step % 100 == 0:
                        brownian_bridge_net.eval()
                        test_X, test_X_cond = next(iter(test_loader))
                        test_X = test_X.to(self.config.device)
                        test_X_cond = test_X_cond.to(self.config.device)

                        test_X_quant, _, _ = vqmodel_ori.encode(test_X)
                        test_X_cond_quant, _, _ =  vqmodel_cond.encode(test_X_cond)

                        test_loss = brownian_bridge_net(test_X_quant, test_X_cond_quant)
                        self.writer.add_scalar(f'vqbb/test', test_loss, step)

                    if step % 5000 == 0:
                        states = [
                            brownian_bridge_net.state_dict(),
                            optimizer.state_dict(),
                            scheduler.state_dict()
                        ]
                        torch.save(states, os.path.join(self.args.model_path, f'vqbb_{step}.pth'))
                        torch.save(states, os.path.join(self.args.model_path, f'vqbb.pth'))
                except BaseException as e:
                    print('Exception save model start!!!')
                    states = [
                        brownian_bridge_net.state_dict(),
                        optimizer.state_dict(),
                        scheduler.state_dict()
                    ]
                    torch.save(states, os.path.join(self.args.model_path, f'vqbb_exp.pth'))
                    torch.save(states, os.path.join(self.args.model_path, f'vqbb.pth'))
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
            scheduler.state_dict()
        ]
        torch.save(states, os.path.join(self.args.model_path, f'vqbb_{step}.pth'))
        torch.save(states, os.path.join(self.args.model_path, f'vqbb.pth'))

    def train(self):
        # self.train_vqmodel(type='ori')
        # self.train_vqmodel(type='cond')
        self.train_vqgan(type='ori')
        self.train_vqgan(type='cond')
        self.train_VQBB()

    @torch.no_grad()
    def test(self):
        vqmodel_config = get_vqmodel_config_dict(self.config.vqmodel.model)
        vqmodel_ori = VQModel(**vqmodel_config).to(self.config.device)
        states = torch.load(os.path.join(self.args.model_path, f'vqmodel_ori.pth'))
        vqmodel_ori.load_state_dict(states[0])
        vqmodel_ori.requires_grad_(False)
        vqmodel_ori.eval()

        vqmodel_cond = VQModel(**vqmodel_config).to(self.config.device)
        states = torch.load(os.path.join(self.args.model_path, f'vqmodel_cond.pth'))
        vqmodel_cond.load_state_dict(states[0])
        vqmodel_cond.requires_grad_(False)
        vqmodel_cond.eval()

        if self.config.vqbbmodel.model.name == 'BB':
            brownian_bridge_net = BrownianBridgeNet(self.config.vqbbmodel).to(self.config.device)
        if self.config.vqbbmodel.model.name == 'BB_2':
            brownian_bridge_net = BrownianBridgeNet_2(self.config.vqbbmodel).to(self.config.device)
        elif self.config.vqbbmodel.model.name == 'CDBB':
            brownian_bridge_net = CDiffuseBrownianBridgeNet(self.config.vqbbmodel).to(self.config.device)
        elif self.config.vqbbmodel.model.name == 'CDBB_2':
            brownian_bridge_net = CDiffuseBrownianBridgeNet_2(self.config.vqbbmodel).to(self.config.device)

        states = torch.load(os.path.join(self.args.model_path, f'vqbb.pth'))
        brownian_bridge_net.load_state_dict(states[0])
        brownian_bridge_net.requires_grad_(False)
        brownian_bridge_net.eval()

        train_dataset, test_dataset = get_dataset(self.config.data)
        train_loader = DataLoader(train_dataset, batch_size=self.config.vqbbmodel.training.batch_size, shuffle=True,
                                  num_workers=8, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.vqbbmodel.training.batch_size, shuffle=True,
                                 num_workers=8, drop_last=True)

        def sample(X, X_cond):
            X = X.to(self.config.device)
            X_cond = X_cond.to(self.config.device)
            # pdb.set_trace()
            X_cond_quant, _, _ = vqmodel_cond.encode(X_cond)
            mid = brownian_bridge_net.p_sample_loop(X_cond_quant)
            mid_quant, _, _ = vqmodel_ori.quantize(mid[-1])
            out = vqmodel_ori.decode(mid_quant)
            return out

        for i in range(5):
            print(i)
            sample_path = os.path.join(self.args.image_path, str(i))
            mkdir(sample_path)
            train_X, train_X_cond = next(iter(train_loader))
            out = sample(train_X, train_X_cond)
            self.save_image(train_X, os.path.join(sample_path, 'train'), 'ground_truth.png')
            self.save_image(train_X_cond, os.path.join(sample_path, 'train'), 'img_cond.png')
            self.save_image(out, os.path.join(sample_path, 'train'), 'img_out.png')

            test_X, test_X_cond = next(iter(test_loader))
            out = sample(test_X, test_X_cond)
            self.save_image(test_X, os.path.join(sample_path, 'test'), 'ground_truth.png')
            self.save_image(test_X_cond, os.path.join(sample_path, 'test'), 'img_cond.png')
            self.save_image(out, os.path.join(sample_path, 'test'), 'img_rec.png')
        return

