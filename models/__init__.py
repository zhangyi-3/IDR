import importlib
import os.path as osp
import os

from utils.registry import ARCH_REGISTRY

arch_folder = osp.dirname(osp.abspath(__file__))

arch_filenames = [item.replace('.py', '') for item in os.listdir(arch_folder) if '__init' not in item]

# import all the arch modules
_arch_modules = [importlib.import_module(f'models.{file_name}') for file_name in arch_filenames]

__all__ = ['build_model']


def build_model(cfg):
    model_name = cfg.model_name

    if type(cfg.get('model_args', ' ')) == str:
        return ARCH_REGISTRY.get(model_name)()
    else:
        return ARCH_REGISTRY.get(model_name)(**cfg.model_args)
