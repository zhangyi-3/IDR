import argparse

import torch.nn.functional as F
# from skimage.measure import compare_psnr, compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from utils import *
# from models import UNet_n2n_un
from models import build_model


def model_forward(net, noisy, padding=32):
    h, w, _ = noisy.shape
    pw, ph = (w + 31) // 32 * 32 - w, (h + 31) // 32 * 32 - h
    with torch.no_grad():
        input_var = torch.FloatTensor([noisy]).cuda().permute(0, 3, 1, 2)
        input_var = F.pad(input_var, (0, pw, 0, ph), mode='reflect')
        # print(input_var.shape,  noisy.shape, ph, pw)
        out_var = net(input_var)

    if pw != 0:
        out_var = out_var[..., :, :-pw]
    if ph != 0:
        out_var = out_var[..., :-ph, :]

    denoised = out_var.permute([0, 2, 3, 1])[0].detach().cpu().numpy()
    return denoised


def add_noise(clean, ntype, sigma=None):
    # assert ntype.lower() in ['gaussian', 'gaussian_gray', 'impulse', 'binomial', 'pattern1', 'pattern2', 'pattern3', 'line']
    assert sigma < 1
    if 'gaussian' in ntype:
        noisy = clean + np.random.normal(0, sigma, clean.shape)

    elif ntype == 'binomial':
        h, w, c = clean.shape
        mask = np.random.binomial(n=1, p=(1 - sigma), size=(h, w, 1))
        noisy = clean * mask

    elif ntype == 'impulse':
        mask = np.random.binomial(n=1, p=(1 - sigma), size=clean.shape)
        noisy = clean * mask

    elif ntype[:4] == 'line':
        # sigma = 25 / 255.0
        h, w, c = clean.shape
        line_noise = np.ones_like(clean) * np.random.normal(0, sigma, (h, 1, 1))
        noisy = clean + line_noise

    elif ntype[:7] == 'pattern':
        # sigma = 5 / 255.0
        h, w, c = clean.shape
        n_type = int(ntype[7:])

        one_image_noise, _, _ = get_experiment_noise('g%d' % n_type, sigma, 0, (h, w, 3))
        noisy = clean + one_image_noise
    else:
        assert 'not support %s' % args.ntype

    return noisy


def test(args, net, test_data_path_set):
    for test_data_path in test_data_path_set:
        data_list = [os.path.join(test_data_path, item) for item in os.listdir(test_data_path) if
                     'jpg' in item or 'png' in item]

        for noise_level in args.test_noise_levels:
            if args.save_img:
                save_dir = os.path.join(args.res_dir, '%s_n' % (args.ntype), 'sigma-%d' % (noise_level))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)

            res = {'psnr': [], 'ssim': []}
            for idx, item in enumerate(data_list):
                gt = cv2.imread(item)
                if 'gray' in args.ntype:
                    gt = cv2.imread(item, 0)[..., np.newaxis]

                gt_ = gt.astype(float) / 255.0
                sigma = noise_level / 255. if noise_level > 1 else noise_level

                noisy = add_noise(gt_, args.ntype, sigma=sigma)

                if args.zero_mean:
                    noisy = noisy - 0.5

                print('\rprocess', idx, len(data_list), item.split('/')[-1], gt.shape, args.ntype, end='')
                denoised = model_forward(net, noisy)

                denoised = denoised + (0.5 if args.zero_mean else 0)
                denoised = np.clip(denoised * 255.0 + 0.5, 0, 255).astype(np.uint8)

                noisy = noisy + (0.5 if args.zero_mean else 0)
                noisy = np.clip(noisy * 255.0 + 0.5, 0, 255).astype(np.uint8)

                # save PSNR
                temp_psnr = compare_psnr(denoised, gt, data_range=255)
                temp_ssim = compare_ssim(denoised, gt, data_range=255, multichannel=True)

                res['psnr'].append(temp_psnr)
                res['ssim'].append(temp_ssim)

                if args.save_img:
                    filename = item.split('/')[-1].split('.')[0] + '_%s' % args.ntype

                    cv2.imwrite(os.path.join(save_dir, '%s_%.2f_out.png' % (filename, temp_psnr)), denoised)
                    cv2.imwrite(os.path.join(save_dir, '%s_NOISY.png' % (filename)), noisy)
                    cv2.imwrite(os.path.join(save_dir, '%s_GT.png' % (filename)), gt)

            print('\r', 'noise lelvel', noise_level, test_data_path.split('/')[-1], len(data_list),
                  ', psnr  %.2f ssim %.3f' % (np.mean(res['psnr']), np.mean(res['ssim'])), args.ntype)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='self supervised')
    parser.add_argument('--root', default="/mnt/lustre/share/cp/zhangyi3/", type=str)

    parser.add_argument('--ntype', default="gaussian", type=str, help='noise type')
    parser.add_argument('--model_path', default=None, type=str)

    parser.add_argument('--res_dir', default="results", type=str)
    parser.add_argument('--save_img', default=True, type=bool)
    args = parser.parse_args()

    args.zero_mean = False
    if args.model_path is None:
        args.model_path = 'checkpoint/%s.pth' % args.ntype

    print('Testing', args.model_path)

    # set testing noise levels
    if "gaussian" in args.ntype:
        args.zero_mean = True
        args.test_noise_levels = [25, 50]
    elif args.ntype == 'line':
        args.test_noise_levels = [25]
    elif args.ntype in ['binomial', 'impulse']:
        args.test_noise_levels = [0.5]
    else:
        args.test_noise_levels = [5]

    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)

    # set testing datasets
    if 'gray' in args.ntype:
        test_data_path_set = [args.root + 'BSD68']
    else:
        test_data_path_set = [args.root + 'kodak',
                              args.root + 'BSDS300/all', ]

    # model
    ch = 1 if 'gray' in args.ntype else 3
    # net = UNet_n2n_un(in_channels=ch, out_channels=ch)
    cfg = EasyDict()
    cfg.model_name = 'UNet_n2n_un'
    cfg.model_args = {'in_channels': ch, 'out_channels': ch}
    net = build_model(cfg)

    net = torch.nn.DataParallel(net)
    net = net.cuda()
    net.load_state_dict(torch.load(args.model_path))
    net.eval()

    test(args, net, test_data_path_set)
