import os.path as osp
import importlib

from torch.utils.data import DataLoader

from utils import scandir
from utils.registry import DATASET_REGISTRY

data_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [
    osp.splitext(osp.basename(v))[0] for v in scandir(data_folder) if '__init__' not in v
    # if v.endswith('_dataset.py')
]
# import all the dataset modules
_dataset_modules = [
    importlib.import_module(f'dataset.{file_name}')
    for file_name in dataset_filenames
]

from functools import partial
import numpy  as np
import random


def worker_init_fn(worker_id, num_workers, rank, seed):
    # Set the worker seed to num_workers * rank + worker_id + seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataset(cfg):
    train_dataset = DATASET_REGISTRY.get(cfg.train_dataset)(**cfg.dataset.train)
    train_sampler = DATASET_REGISTRY.get('EnlargedSampler')(train_dataset, cfg.world_size, cfg.rank,
                                                            cfg.dataset.train.sample_ratio)
    if cfg.dataset.train.total_bs > 0:
        assert cfg.dataset.train.total_bs % cfg.world_size == 0
        cfg.dataset.train.batch_size_per_gpu = int(cfg.dataset.train.total_bs / cfg.world_size)
        if cfg.rank == 0:
            print('** reset per gpu bs:', cfg.dataset.train.batch_size_per_gpu, 'ws', cfg.world_size,
                  cfg.dataset.train.total_bs)
    if not cfg.dist:
        cfg.dataset.train.batch_size_per_gpu = cfg.dataset.train.total_bs

    train_loader = DataLoader(train_dataset, batch_size=cfg.dataset.train.batch_size_per_gpu, shuffle=False,
                              num_workers=cfg.dataset.train.worker_num, pin_memory=True, sampler=train_sampler,
                              drop_last=True, persistent_workers=True,
                              worker_init_fn=partial(worker_init_fn, num_workers=cfg.dataset.train.worker_num,
                                                     rank=cfg.rank, seed=cfg.seed))

    if cfg.dataset.train.total_bs > 0 and cfg.rank == 0:
        print('end dataloader before prefetcher; iters per epoch:', len(train_loader), len(train_dataset))
        # assert len(train_loader) >= cfg.total_iter  # ensure one epoch training
    prefetcher = DATASET_REGISTRY.get(cfg.prefetcher)(train_loader)
    return prefetcher, train_sampler
