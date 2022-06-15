from .synthetic_config import cfgs as syn_cfg
from utils import EasyDict

# from .bj_sidd_config import cfgs as bj_sidd_cfg

# get cluster name
import socket, os
import torch

hostname = socket.gethostname()
# if int(os.environ.get('SLURM_PROCID', default=0)) == 0:
#     try:
#         pass
#         print('*** cluster ', hostname, torch.cuda.get_device_properties(0).total_memory)
#     except:
#         pass

all_config = EasyDict()
for subconfig in [syn_cfg]:
    for key, value in subconfig.items():
        all_config[key] = EasyDict(value)

        # modify configs for different clusters
        cfg = all_config[key]
        try:
            if '10-10-16' in hostname:
                cfg.dataset.train.path = cfg.dataset.train.path.replace('/lustre/', '/lustreold/')
                cfg.dataset.train.meta_file = cfg.dataset.train.meta_file.replace('/lustre/', '/lustreold/')
            if '10-142-4' in hostname:
                if os.path.exists('/mnt/cache/share/images/'):
                    cfg.dataset.train.path = cfg.dataset.train.path.replace('/mnt/lustre/share/images/',
                                                                            '/mnt/cache/share/images/')
                    cfg.dataset.train.meta_file = cfg.dataset.train.meta_file.replace('/mnt/lustre/share/images/',
                                                                                      '/mnt/cache/share/images/')
                else:
                    cfg.dataset.train.path = cfg.dataset.train.path.replace('/share/images/',
                                                                            '/share/zhangyi3/imagenet/')
                    cfg.dataset.train.meta_file = cfg.dataset.train.meta_file.replace('/share/images/',
                                                                                      '/share/zhangyi3/imagenet/')
        except:
            pass
        all_config[key] = cfg
