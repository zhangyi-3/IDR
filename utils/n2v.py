import torch

import numpy as np

def __get_stratified_coords2D__(perc_pix = 1.5, shape=(256, 256)):
    box_size = np.round(np.sqrt(100 / perc_pix)).astype(np.int)

    def __rand_float_coords2D__(boxsize=box_size):
        while True:
            yield (np.random.rand() * boxsize, np.random.rand() * boxsize)

    coord_gen = __rand_float_coords2D__()

    box_count_y = int(np.ceil(shape[0] / box_size))
    box_count_x = int(np.ceil(shape[1] / box_size))
    x_coords = []
    y_coords = []
    for i in range(box_count_y):
        for j in range(box_count_x):
            y, x = next(coord_gen)
            y = int(i * box_size + y)
            x = int(j * box_size + x)
            if (y < shape[0] and x < shape[1]):
                y_coords.append(y)
                x_coords.append(x)
    return (y_coords, x_coords)


def get_subpatch(patch, coord, local_sub_patch_radius=5):
    start = np.maximum(0, np.array(coord) - local_sub_patch_radius)
    end = start + local_sub_patch_radius*2 + 1

    shift = np.minimum(0, patch.shape - end)

    start += shift
    end += shift

    slices = [ slice(s, e) for s, e in zip(start, end)]

    return patch[tuple(slices)]

def pm_uniform_withCP(local_sub_patch_radius=5):
    def random_neighbor_withCP_uniform(patch, coords, dims=2):
        vals = []
        for coord in zip(*coords):
            sub_patch = get_subpatch(patch, coord,local_sub_patch_radius)
            rand_coords = [np.random.randint(0, s) for s in sub_patch.shape[0:dims]]
            vals.append(sub_patch[tuple(rand_coords)])
        return vals
    return random_neighbor_withCP_uniform

def generate_n2v_mask(input, perc_pix=1.5, local_sub_patch_radius=5):
    b, c, h, w = input.shape

    if type(input) == np.ndarray:
        output = input.copy()
        mask = np.zeros_like(input)
    elif type(input) == torch.Tensor:
        output = input.clone()
        if input.is_cuda:
            mask = torch.cuda.FloatTensor(input.shape).fill_(0)
        else:
            mask = torch.FloatTensor(input.shape).fill_(0)
    else:
        assert 'not support'

    box_size = np.round(np.sqrt(100 / perc_pix)).astype(np.int)
    assert h % box_size == 0 and w % box_size == 0
    box_count_y = h // box_size
    box_count_x = w // box_size

    coord = np.random.randint(0, box_size, (b, c, box_count_x, box_count_y, 2))
    (x_shift, y_shift) = np.meshgrid(np.arange(box_count_x) * box_size, np.arange(box_count_y)*box_size)
    coord[..., 0] += x_shift
    coord[..., 1] += y_shift

    mask_coord = coord.reshape((-1 ,2))
    num_sample = box_count_x * box_count_y

    start = np.maximum(0, mask_coord - local_sub_patch_radius)
    end = start + local_sub_patch_radius*2 + 1

    shift = np.minimum(0, (h, w) - end)

    start += shift
    msk_neigh_coord = start + np.random.randint(0, local_sub_patch_radius * 2 + 1, start.shape)

    B, C, _ = np.meshgrid(range(b), range(c), range(num_sample))
    # print(B.shape, C.shape, idx_msk.shape, idy_msk.shape, idx_msk_neigh.shape)
    id_msk = (B.flatten(), C.flatten(),  mask_coord[..., 0], mask_coord[..., 1])
    id_msk_neigh = (B.flatten(), C.flatten(), msk_neigh_coord[..., 0], msk_neigh_coord[..., 1])
    # print(output[id_msk].shape, input[id_msk_neigh].shape)
    output[id_msk] = input[id_msk_neigh]
    mask[id_msk] = 1.0

    # DEBUG = True
    # if DEBUG:
    #     import matplotlib.pyplot as plt
    #     plt.imshow(mask[np.random.randint(mask.shape[0]), np.random.randint(mask.shape[1])] * 255)
    #     plt.show()
    #     plt.close()
    return output, mask

def generate_naive_mask(input, ratio=0.015, wins=[5, 5]):
    b, c, h, w = input.shape
    num_sample = int(h * w * ratio)

    if type(input) == np.ndarray:
        output = input.copy()
        mask = np.zeros_like(input)
    elif type(input) == torch.Tensor:
        output = input.clone()
        if input.is_cuda:
            mask = torch.cuda.FloatTensor(input.shape).fill_(0)
        else:
            mask = torch.FloatTensor(input.shape).fill_(0)
    else:
        assert 'not support'

    wh, ww = wins[0] // 2, wins[1] // 2

    # sample center point
    idy_msk = np.random.randint(0 + wh, h - wh, (b, c, num_sample))
    idx_msk = np.random.randint(0 + ww, w - ww, (b, c, num_sample))

    idy_neigh = np.random.choice(list(range(1, wh + 1)) + list(range(-wh, 0)), (b, c, num_sample))
    idx_neigh = np.random.choice(list(range(1, ww + 1)) + list(range(-ww, 0)), (b, c, num_sample))

    idy_msk_neigh = idy_msk + idy_neigh
    idx_msk_neigh = idx_msk + idx_neigh

    B, C, _ = np.meshgrid(range(b), range(c), range(num_sample))

    id_msk = (B.flatten(), C.flatten(),  idy_msk.flatten(), idx_msk.flatten())
    id_msk_neigh = (B.flatten(), C.flatten(), idy_msk_neigh.flatten(), idx_msk_neigh.flatten())
    output[id_msk] = input[id_msk_neigh]
    mask[id_msk] = 1.0
    return output, mask

if __name__ == '__main__':
    inter_val = eval('pm_uniform_withCP()')
    coord = __get_stratified_coords2D__(perc_pix=1.5, shape=(256, 256))
    x_val = inter_val(np.random.random((256, 256)), coord)

    a, b, c = np.meshgrid(range(3), range(4), range(5))
    generate_n2v_mask(np.random.random((32, 4, 256, 256)), 1.5)
