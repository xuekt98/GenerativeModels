import argparse

import yaml
import io

# s = {'tuple': (1, 2, 3, 4)}

# with io.open('test.yml', 'w', encoding='utf-8') as wf:
#    yaml.dump(s, wf)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


with open('DDPM.yml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
new_config = dict2namespace(config)
print(type(new_config.model.dim_mults))