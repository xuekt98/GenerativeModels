import argparse
import os
import yaml
import torch
from runners.NCSN_runner import NCSNRunner
from runners.DDPM_runner import DDPMRunner
from runners.DDGM_runner import DDGMRunner
from runners.CDIFFUSE_runner import CDIFFUSERunner


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--runner', type=str, default='NCSNRunner', help='The runner to execute')
    parser.add_argument('--config', type=str, default='NCSN.yml', help='Path to the config file')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('-o', '--output_path', type=str, default='output', help="The directory of image outputs")
    parser.add_argument('--load_model', action='store_true', default=False, help='Whether to resume training')
    parser.add_argument('--load_path', type=str, default='output/2022-02-28T18-46-08', help='Path to the state dict')
    parser.add_argument('--load_iter', type=int, default=-1, help='Path to the state dict')
    parser.add_argument('--test', action='store_true', default=False, help='Whether to test the model')

    args = parser.parse_args()

    with open(os.path.join("configs", args.config), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    new_config = dict2namespace(config)

    new_config.device = torch.device('cuda:0')

    return args, new_config


def main():
    args, config = parse_args_and_config()
    #try:
    runner = eval(args.runner)(args, config)
    if not args.test:
        runner.train()
    else:
        runner.test()
    # except BaseException as e:
    #     print('Exception save model start!!!')
    #     print(e)
    return


if __name__ == "__main__":
    main()
