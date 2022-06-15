import argparse
import torch
import os
import yaml

import numpy as np

from utils import set_seed_dist, check_empty_dir
from utils.dist_util import init_dist, get_dist_info

from trainer import build_trainer
from dataset import build_dataset
from configs import all_config

assert torch.__version__.split('.') != '0'


def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n', type=str, required=True, help='Config name')
    parser.add_argument(
        '-launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='slurm',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    cfg = all_config[args.n]
    cfg.update(args.__dict__)
    # opt = parse(cfg.opt, is_train=is_train)

    # distributed settings
    if cfg.launcher == 'none':
        cfg.dist = False
        print('Disable distributed.', flush=True)
    else:
        cfg.dist = True
        if cfg.launcher == 'slurm' and 'dist_params' in cfg.keys():
            init_dist(cfg.launcher, **cfg.dist_params)
        else:
            init_dist(cfg.launcher)

    cfg.rank, cfg.world_size = get_dist_info()
    # print('rank', cfg.rank, 'ws', cfg.world_size)

    # random seed for dist training
    if cfg.seed is None:
        cfg.seed = np.random.randint(1e8)
    temp_seed = set_seed_dist(cfg.seed + cfg.rank)
    torch.backends.cudnn.benchmark = False  # efficient training
    torch.backends.cudnn.deterministic = True

    if cfg.rank == 0:
        print('set seed', temp_seed, cfg.seed, cfg.rank)
        cfg.seed = temp_seed
        print(cfg)

    return cfg


def train_iters(model):
    while True:
        train_sampler.set_epoch(model.epoch)

        prefetcher.reset()
        data = prefetcher.next()

        while data is not None:

            model.preprocess(data)
            model.optimize_parameters()
            model.update_lr_scheme()

            data = prefetcher.next()

            if model.cur_iter >= cfg.total_iter:
                if cfg.rank == 0:
                    print('End training, Save the latest epoch model, total iterations', cfg.total_iter)
                exit(0)
        model.epoch += 1


if __name__ == '__main__':
    cfg = parse_options()
    if cfg.rank == 0: print('** build trainer', cfg.trainer.name)
    model = build_trainer(cfg)

    if cfg.rank == 0:  print('** end dataloader', cfg.train_dataset)
    prefetcher, train_sampler = build_dataset(cfg)
    if cfg.rank == 0:  print('** end prefetcher')

    if cfg.rank == 0:
        # create dir before training
        assert cfg.model_dir is not None
        log_path = os.path.join('log-trans', cfg.name)
        for temp_dir in [cfg.model_dir, cfg.result_dir, log_path]:
            check_empty_dir(temp_dir)

        # Save current cfg
        with open(os.path.join(cfg.model_dir, 'cfg.yaml'), 'w') as f:
            yaml.dump(cfg.copy(), f)  # copy() it or it saves nothing.

    if cfg.rank == 0: print('Epoch start')
    train_iters(model)
