import os
import cv2
import time
import io

import numpy as np

from torch.utils.data import Dataset
from PIL import Image
from utils.registry import DATASET_REGISTRY

try:
    import mc
except:
    print('** import mc error')
    pass


def random_crop(img, size):
    ''' only support input HWC image with square crops
    '''
    h, w, _ = img.shape
    if w <= size or h <= size:  # resize too small images
        img = cv2.resize(img, (size, size))
    else:
        hs, ws = np.random.randint(h - size), np.random.randint(w - size)
        img = img[hs:hs + size, ws:ws + size]
    return img


def augment_img(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def pil_loader(img_str):
    buff = io.BytesIO(img_str)

    with Image.open(buff) as img:
        img = img.convert('RGB')
    return img


class MemcachedBase(Dataset):
    def __init__(self):
        super(MemcachedBase, self).__init__()
        self.initialized = False

    def _init_memcached(self):
        if not self.initialized:
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/zhangyi3/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def _mem_load(self, filename):
        self._init_memcached()
        value = mc.pyvector()
        self.mclient.Get(filename, value)
        value_str = mc.ConvertBuffer(value)
        img = pil_loader(value_str)
        return img


@DATASET_REGISTRY.register()
class Imagefolder(MemcachedBase):
    def __init__(self, path, **kwargs):
        self.root_dir = path  # root_dir
        self.size = kwargs['crop_size']
        self.repeat = int(kwargs['repeat'])
        self.mc = kwargs['mc']
        self.aug = kwargs['aug']
        self.preload = kwargs['preload']
        self.img_range = kwargs['img_range']

        self.rank = int(os.environ.get('SLURM_PROCID', default=0))

        # self.rank = link.get_rank()
        def check_img_type(item):
            return '.jpg' in item.lower() or '.png' in item.lower() or '.bmp' in item.lower() or '.JPEG'

        if isinstance(self.root_dir, list):
            lines = []
            for one_dir in self.root_dir:
                lines += [os.path.join(one_dir, item) for item in os.listdir(one_dir) if check_img_type(item)]

        else:
            lines = [os.path.join(self.root_dir, item) for item in os.listdir(self.root_dir) if check_img_type(item)]
        assert len(lines) > 0

        if self.rank == 0:
            print("load from folder %s num: %d\n%s" % (self.root_dir, len(lines), lines[0]), 'repeat', self.repeat)

        self.num = len(lines)
        self.metas = lines

        if self.preload:
            self.imgs = [None] * self.num
            ## todo multiple threads loading
            start = time.time()

            for idx, item in enumerate(lines):
                if self.rank == 0 and idx % 50 == 0:
                    print('\rpreloading %d / %d, time %.1f' % (idx, self.num, (time.time() - start) / 60), end='')
                bgr = cv2.imread(os.path.join(self.root_dir, item))
                img = bgr[..., ::-1]
                self.imgs[idx] = img

            def load_img(idx, path):
                return cv2.imread(path)[..., ::-1]

            # self.imgs = utils.multi_run(lines, load_img, type='thread')
            if self.rank == 0:
                print('dataset preloading time cost %.1f' % ((time.time() - start) / 60))

        self.initialized = False

    def __len__(self):
        return self.num * self.repeat

    def __getitem__(self, idx):
        idx = idx // self.repeat
        filename = self.metas[idx]

        if self.preload:
            img = self.imgs[idx]
        elif not self.mc:
            bgr = cv2.imread(filename)
            img = bgr[..., ::-1].astype(float)
        else:
            # cls = self.metas[idx][1]
            ## memcached
            img = self._mem_load(filename)

        # random crop 256
        img = np.array(img, dtype=float)
        img = random_crop(img, self.size)

        if self.aug:
            mode = np.random.randint(8)
            img = augment_img(img, mode)

        # image range
        if self.img_range == 1:
            img = img / 255.

        return {'gt': img.astype(np.float32)}
