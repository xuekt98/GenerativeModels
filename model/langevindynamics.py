import numpy as np
import torch
import tqdm
import pdb


def Langevin_dynamics(x_mod, scorenet, n_steps=200, step_lr=0.00005):
    images = []
    labels = torch.ones(x_mod.shape[0], device=x_mod.device) * 9
    labels = labels.long()
    with torch.no_grad():
        for _ in range(n_steps):
            images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
            noise = torch.randn_like(x_mod) * np.sqrt(step_lr * 2)
            grad = scorenet(x_mod, labels)
            x_mod = x_mod + step_lr * grad + noise
            x_mod = x_mod
            print("modulus of grad components: mean {}, max {}".format(grad.abs().mean(), grad.abs().max()))

        return images


def anneal_Langevin_dynamics(x_mod, scorenet, sigmas, n_steps_each=100, step_lr=0.00002):
    images = []

    with torch.no_grad():
        for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc='annealed Langevin dynamics sampling'):
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2
            for s in range(n_steps_each):
                images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
                noise = (torch.randn_like(x_mod)) * np.sqrt(step_size.to('cpu') * 2)
                grad = scorenet(x_mod, labels)
                x_mod = x_mod + step_size * grad + noise

        return images