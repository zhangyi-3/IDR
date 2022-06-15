import importlib
import os.path as osp

from copy import deepcopy

from utils import scandir
from utils.registry import TRAINER_REGISTRY

arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder) if v.endswith('_trainer.py')]
# print(arch_filenames, __file__)
# import all the arch modules
_arch_modules = [importlib.import_module(f'trainer.{file_name}') for file_name in arch_filenames]

__all__ = ['build_trainer']


def build_trainer(cfg):
    cfg = deepcopy(cfg)

    trainer = TRAINER_REGISTRY.get(cfg.trainer.name)(cfg)
    # logger = get_root_logger()
    # logger.info(f'Network [{net.__class__.__name__}] is created.')
    return trainer
