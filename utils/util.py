from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from scipy.signal import fftconvolve
from bm3d import gaussian_kernel

import torch
import random, shutil, datetime
import os, cv2, traceback, math
import os.path as osp
import numpy as np

from multiprocessing.pool import ThreadPool


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


def basic_loss(img, truth, type='tv'):
    if type == 'l1':
        return torch.mean(torch.abs(img - truth))
    elif type == 'l2':
        return torch.mean((img - truth) ** 2)
    elif type == 'psnr':

        return 10 / np.log(10) * torch.log(((img - truth) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
    else:
        assert 'not support'


def save_jpg(path, img, quality=100):
    assert path[-4:] == '.jpg'
    cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return


def save_img(path, save_name, input_var, output, gt=None, rgb=False):
    SRGB = True if input_var.shape[1] == 3 else False
    if SRGB and input_var.mean() > 10:  # for number scale 255
        input_var, output = input_var / 255., output / 255.
        if gt is not None:
            gt = gt / 255.
    try:
        if not SRGB:  # raw
            idx = min(2, input_var.shape[1]) - 1
            input_np = input_var[0, idx].detach().cpu().numpy().clip(0, 1)  # green channel
            output_np = output[0, idx].detach().cpu().numpy().clip(0, 1)
        else:
            input_np = input_var.permute(0, 2, 3, 1)[0].detach().cpu().numpy().clip(0, 1)
            output_np = output.permute(0, 2, 3, 1)[0].detach().cpu().numpy().clip(0, 1)

        if input_np.shape != output_np.shape:
            temp = np.zeros_like(output_np)
            ih, iw, _ = input_np.shape
            temp[:ih, :iw] = input_np
            input_np = temp

        if gt is not None:
            if SRGB:
                gt_np = gt.permute(0, 2, 3, 1)[0].detach().cpu().numpy().clip(0, 1)
            else:
                idx = min(2, gt.shape[1]) - 1
                gt_np = gt[0, idx].detach().cpu().numpy().clip(0, 1)
            img = np.concatenate([input_np, output_np, gt_np], axis=1)
        else:
            img = np.concatenate([input_np, output_np], axis=1)

        # normalization
        if img.mean() < 0.1 and not SRGB:
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)

        if img.mean() < 0.05:
            img = img ** (1 / 3)
            # _, h, w = img.shape
        # img = img.reshape((2, 8, h, w)).transpose(1, 2, 3 ,0)
        # cv2.imwrite(os.path.join(path, save_name), img.reshape((h * 8, w * 2))*255)

        # cv2.imwrite(os.path.join(path, save_name), img * 255)
        if rgb:
            img = img[..., ::-1]
        save_jpg(os.path.join(path, save_name), img * 255, quality=100)
    except:
        print(print(traceback.format_exc()))
        print('srgb?', SRGB, 'gt is None?', gt is None)
        print('var shape', input_var.shape, output.shape)
        print('np shape', input_np.shape, output_np.shape)
        if gt is not None:
            print(gt.shape, gt_np.shape)
    return


def get_psnr(y_est: np.ndarray, y_ref: np.ndarray) -> float:
    """
    Return PSNR value for y_est and y_ref presuming the noise-free maximum is 1.
    :param y_est: Estimate array
    :param y_ref: Noise-free reference
    :return: PSNR value
    """
    return 10 * np.log10(1 / np.mean(((y_est - y_ref).ravel()) ** 2))


def get_cropped_psnr(y_est: np.ndarray, y_ref: np.ndarray, crop: tuple) -> float:
    """
    Return PSNR value for y_est and y_ref presuming the noise-free maximum is 1.
    Crop the images before calculating the value by crop.
    :param y_est: Estimate array
    :param y_ref: Noise-free reference
    :param crop: Tuple of crop-x and crop-y from both stides
    :return: PSNR value
    """
    return get_psnr(np.atleast_3d(y_est)[crop[0]:-crop[0], crop[1]:-crop[1], :],
                    np.atleast_3d(y_ref)[crop[0]:-crop[0], crop[1]:-crop[1], :])


def get_experiment_kernel(noise_type: str, noise_var: float, sz: tuple = np.array((101, 101))):
    """
    Get kernel for generating noise from specific experiment from the paper.
    :param noise_type: Noise type string, g[0-4](w|)
    :param noise_var: noise variance
    :param sz: size of image, used only for g4 and g4w
    :return: experiment kernel with the l2-norm equal to variance
    """
    # if noiseType == gw / g0
    kernel = np.array([[1]])
    noise_types = ['gw', 'g0', 'g1', 'g2', 'g3', 'g4', 'g1w', 'g2w', 'g3w', 'g4w']
    if noise_type not in noise_types:
        raise ValueError("Noise type must be one of " + str(noise_types))

    if noise_type != "g4" and noise_type != "g4w":
        # Crop this size of kernel when generating,
        # unless pink noise, in which
        # if noiseType == we want to use the full image size
        sz = np.array([101, 101])
    else:
        sz = np.array(sz)

    # Sizes for meshgrids
    sz2 = -(1 - (sz % 2)) * 1 + np.floor(sz / 2)
    sz1 = np.floor(sz / 2)
    uu, vv = np.meshgrid([i for i in range(-int(sz1[0]), int(sz2[0]) + 1)],
                         [i for i in range(-int(sz1[1]), int(sz2[1]) + 1)])

    beta = 0.8

    if noise_type[0:2] == 'g1':
        # Horizontal line
        kernel = np.atleast_2d(16 - abs(np.linspace(1, 31, 31) - 16))

    elif noise_type[0:2] == 'g2':
        # Circular repeating pattern
        scale = 1
        dist = uu ** 2 + vv ** 2
        kernel = np.cos(np.sqrt(dist) / scale) * gaussian_kernel((sz[0], sz[1]), 10)

    elif noise_type[0:2] == 'g3':
        # Diagonal line pattern kernel
        scale = 1
        kernel = np.cos((uu + vv) / scale) * gaussian_kernel((sz[0], sz[1]), 10)

    elif noise_type[0:2] == 'g4':
        # Pink noise
        dist = uu ** 2 + vv ** 2
        n = sz[0] * sz[1]
        spec = (np.sqrt((np.sqrt(n) * 1e-2) / (np.sqrt(dist) + np.sqrt(n) * 1e-2)))
        kernel = fftshift(ifft2(ifftshift(spec)))

    else:  # gw and g0 are white
        beta = 0

    # -- Noise with additional white component --

    if len(noise_type) > 2 and noise_type[2] == 'w':
        kernel = kernel / np.sqrt(np.sum(kernel ** 2))
        kalpha = np.sqrt((1 - beta) + beta * abs(fft2(kernel, (sz[0], sz[1]))) ** 2)
        kernel = fftshift(ifft2(kalpha))

    kernel = np.real(kernel)
    # Correct variance
    kernel = kernel / np.sqrt(np.sum(kernel ** 2)) * np.sqrt(noise_var)

    return kernel


# @profile
def get_experiment_noise(noise_type: str, noise_var: float, realization: int, sz: tuple) \
        -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Generate noise for experiment with specified kernel, variance, seed and size.
    Return noise and relevant parameters.
    The generated noise is non-circular.
    :param noise_type: Noise type, see get_experiment_kernel for list of accepted types.
    :param noise_var: Noise variance of the resulting noise
    :param realization: Seed for the noise realization
    :param sz: image size -> size of resulting noise
    :return: noise, PSD, and kernel
    """
    np.random.seed(realization)

    # Get pre-specified kernel
    kernel = get_experiment_kernel(noise_type, noise_var, sz)

    # Create noisy image
    half_kernel = np.ceil(np.array(kernel.shape) / 2)

    if len(sz) == 3 and half_kernel.size == 2:
        half_kernel = [half_kernel[0], half_kernel[1], 0]
        kernel = np.atleast_3d(kernel)

    half_kernel = np.array(half_kernel, dtype=int)

    # Crop edges
    noise = fftconvolve(np.random.normal(size=(sz + 2 * half_kernel)), kernel, mode='same')
    noise = np.atleast_3d(noise)[half_kernel[0]:-half_kernel[0], half_kernel[1]:-half_kernel[1], :]

    psd = abs(fft2(kernel, (sz[0], sz[1]), axes=(0, 1))) ** 2 * sz[0] * sz[1]

    return noise, psd, kernel


def get_experiment_noise_conv(noise_type: str, noise_var: float, realization: int, sz: tuple) \
        -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Generate noise for experiment with specified kernel, variance, seed and size.
    Return noise and relevant parameters.
    The generated noise is non-circular.
    :param noise_type: Noise type, see get_experiment_kernel for list of accepted types.
    :param noise_var: Noise variance of the resulting noise
    :param realization: Seed for the noise realization
    :param sz: image size -> size of resulting noise
    :return: noise, PSD, and kernel
    """
    np.random.seed(realization)

    # Get pre-specified kernel
    kernel = get_experiment_kernel(noise_type, noise_var, sz)

    # Create noisy image
    half_kernel = np.ceil(np.array(kernel.shape) / 2)

    if len(sz) == 3 and half_kernel.size == 2:
        half_kernel = [half_kernel[0], half_kernel[1], 0]
        kernel = np.atleast_3d(kernel)

    half_kernel = np.array(half_kernel, dtype=int)

    # Crop edges
    temp_noise = np.random.normal(size=(sz + 2 * half_kernel))
    noise = fftconvolve(temp_noise, kernel, mode='same')
    noise = np.atleast_3d(noise)[half_kernel[0]:-half_kernel[0], half_kernel[1]:-half_kernel[1], :]

    psd = abs(fft2(kernel, (sz[0], sz[1]), axes=(0, 1))) ** 2 * sz[0] * sz[1]

    return noise, psd, kernel


def add_noise(input_var, cfg, *args):
    noise_type = cfg.noise_type
    noise_level = cfg.noise_level
    assert input_var.is_cuda

    if noise_type == 'g':
        if isinstance(noise_level, list):  # random noise level
            sigma = np.random.uniform(noise_level[0], noise_level[1], size=input_var.shape[0])
            cfg.temp_sigma = 'g' + str(sigma)
            sigma = sigma / 255.0
        else:
            assert NotImplementedError

        if not cfg.mixed_batch_nl:
            noise = torch.cuda.FloatTensor(input_var.shape).normal_(0, sigma[0])
        else:
            noise = torch.cuda.FloatTensor(input_var.shape)
            for idx in range(input_var.shape[0]):
                noise[idx].normal_(0, sigma[idx])
        input_noise = input_var + noise
        return input_noise.float(), sigma, None

    elif noise_type == 'line':
        sigma = noise_level / 255.0
        b, c, h, w = input_var.shape
        line_noise = torch.cuda.FloatTensor(b, 1, h, 1).normal_(0, sigma)
        input_noise = input_var + torch.cuda.FloatTensor(input_var.shape).fill_(1) * line_noise
        return input_noise.float(), sigma, None

    elif noise_type in ['binomial', 'impulse']:
        sigma = noise_level
        b, c, h, w = input_var.shape
        mask_shape = (b, 1, h, w) if noise_type == 'binomial' else (b, c, h, w)
        mask = torch.cuda.FloatTensor(*mask_shape).uniform_(0, 1)
        mask = mask * torch.cuda.FloatTensor(b, 1, 1, 1).uniform_(0, sigma)  # add different noise level for each frame
        mask = 1 - torch.bernoulli(mask)
        input_noise = input_var * mask
        return input_noise.float(), sigma, None

    elif 'scn' in noise_type:  # spatially correlated noise
        sigma = noise_level / 255.0
        b, c, h, w = input_var.shape
        input_noise = input_var.clone()
        n_type = int(noise_type.split('-')[-1])

        def img_add_noise(img, n_type, sigma, h, w):
            one_image_noise, _, _ = get_experiment_noise('g%d' % n_type, sigma, np.random.randint(1e9), (h, w, 3))
            one_image_noise = torch.FloatTensor(one_image_noise).to(img.device).permute(2, 0, 1)  # for dist training
            img = img + one_image_noise
            return img

        pool = ThreadPool(processes=4 if b >= 4 else b)
        result = []
        for i in range(b):  # no need to
            result.append(pool.apply_async(img_add_noise, (input_var[i], n_type, sigma, h, w)))
        pool.close()
        pool.join()
        for i, res in enumerate(result):
            input_noise[i] = res.get()

        return input_noise.float(), sigma, None
    else:
        assert NotImplementedError


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.
    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.
    Returns:
        A generator for all the interested files with relative paths.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def set_seed(manualSeed=1, cuda=False):
    if manualSeed is None:
        manualSeed = np.random.randint(1e8)
    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    if cuda:
        torch.cuda.manual_seed(manualSeed)
        torch.cuda.manual_seed_all(manualSeed)

        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return manualSeed


def set_seed_dist(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def check_empty_dir(log_path, query=False, key='log-'):
    if os.path.exists(log_path):
        pass
    else:
        os.makedirs(log_path)
    return


def create_empty_dir(log_path, query=False, key='log-'):
    if os.path.exists(log_path):

        # if query:
        #     # temp = input('delete and retrain? %s\n (Answer yes/no) : ' % log_path)
        #     # if temp.strip().lower() != 'yes':
        #     #     cprint('%s exists and stop retraining' % log_path)
        #     #     exit(0)
        #     print('** delete existing file')
        #     shutil.rmtree(log_path)
        #     os.makedirs(log_path)

        file_names = [item for item in os.listdir(log_path) if key not in item]
        if len(file_names) == 0:
            return
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        target_dir = os.path.join(log_path, '%s%s' % (key, str(timestamp)))
        os.makedirs(target_dir)
        for idx, file_name in enumerate(file_names):
            print('\rmoving files %d %d' % (idx, len(file_names)), end='')
            shutil.move(os.path.join(log_path, file_name), target_dir)

        # shutil.rmtree(log_path)
    else:
        os.makedirs(log_path)
    return
