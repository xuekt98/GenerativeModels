import torch
import torchvision
import torchvision.transforms as transforms
import os
from datetime import datetime


def mkdir(dir):
    if os.path.exists(dir):
        return
    else:
        os.makedirs(dir)


def make_dirs(args, prefix, now=None):
    if now is None:
        now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    if args.load_model:
        # 如果加载已有的模型，则继续保存在指定的文件夹下
        root_path = os.path.join(args.load_path)
    else:
        root_path = os.path.join(args.output_path, prefix, now)

    log_path = os.path.join(root_path, "log")
    mkdir(log_path)
    model_path = os.path.join(root_path, "model")
    mkdir(model_path)
    image_path = os.path.join(root_path, "images")
    mkdir(image_path)
    sample_path = os.path.join(root_path, "samples")
    mkdir(sample_path)
    return image_path, model_path, log_path, sample_path, now


def get_optimizer(optim_config, parameters):
    if optim_config.optimizer == 'Adam':
        return torch.optim.Adam(parameters, lr=optim_config.lr, weight_decay=optim_config.weight_decay,
                                betas=(optim_config.beta1, 0.999))
    elif optim_config.optimizer == 'RMSProp':
        return torch.optim.RMSprop(parameters, lr=optim_config.lr, weight_decay=optim_config.weight_decay)
    elif optim_config.optimizer == 'SGD':
        return torch.optim.SGD(parameters, lr=optim_config.lr, momentum=0.9)
    else:
        return NotImplementedError('Optimizer {} not understood.'.format(optim_config.optimizer))


def get_dataset(dataset_config):
    trans = transforms.Compose([
        transforms.Resize(dataset_config.image_size),
        transforms.ToTensor()
    ])

    mnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=trans, download=True)
    return mnist_train, mnist_test