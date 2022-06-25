import torch
import time
import os

import numpy as np

from collections import deque
# from tensorboardX import SummaryWriter


from models import build_model
from utils.registry import TRAINER_REGISTRY
from utils import add_noise, save_img, basic_loss


@TRAINER_REGISTRY.register()
class Base_trainer():
    def __init__(self, cfg, train=True):
        self.cfg = cfg

        self.ema_decay = self.cfg.trainer.get('ema_decay', -1)
        self.epoch = 0

        # self.net = self.model_to_device(build_model(cfg))
        self.net = self.model_to_device(torch.nn.SyncBatchNorm.convert_sync_batchnorm(build_model(cfg)))

        if self.ema_decay > 0:
            self.net_ema = build_model(cfg).cuda()
            self.net_ema.eval()

        # optimizer
        self.set_optimizer(cfg)

        self.set_scheduler(cfg)

        # # init log writer and create dir
        # if self.cfg.rank == 0:
        #     self.writer = SummaryWriter(os.path.join('log-trans', cfg.name))

        # loss
        self.lossess = deque(maxlen=50)

        self.s_time = time.time()
        self.pre_used_time = 0
        self.cur_iter = -1

        if self.cfg.resume:
            self.resume_training_state()

        if train:
            self.net.train()

    def test(self):
        pass

    def model_to_device(self, net):
        net = net.cuda()  # move to cuda first, then call distributed data parallel.
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[torch.cuda.current_device()],
                                                        output_device=self.cfg.rank,
                                                        find_unused_parameters=self.cfg.find_unused_params)
        return net

    def set_scheduler(self, cfg):
        if self.cfg.schedule == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.total_iter,
                                                                        **self.cfg.schedule_args)

        return

    def set_optimizer(self, cfg):
        if self.cfg.opt == 'adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

        else:
            raise NotImplementedError
        return

    def preprocess(self, data):
        self.input, self.gt = data['noisy'], data['gt']
        self.meta = data['np']
        # for k in data.keys():
        #     print(k, data[k].shape)

    def optimize_parameters(self):
        # gt, noisy = preprocess(data, cfg)
        # input_var, gt_var, out = model_forward(model, noisy, gt)
        self.optimizer.zero_grad()
        self.output = self.net(self.input)

        loss = basic_loss(self.output, self.gt, type=self.cfg.loss_type)

        loss.backward()

        # psnr = 0  # get_criterion(loss)
        if self.cfg.clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.clip_norm)

        self.optimizer.step()
        self.print_iter_info(loss, 0)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        return loss

    def model_ema(self, decay=0.999):
        if hasattr(self.net, 'module'):
            net_g = self.net.module

        net_g_params = dict(net_g.named_parameters())
        net_ema_params = dict(self.net_ema.named_parameters())

        for k in net_ema_params.keys():
            net_ema_params[k].data.mul_(decay).add_(net_g_params[k].data, alpha=1 - decay)

    def update_lr_scheme(self):
        self.cur_iter += 1
        cur_iter = self.cur_iter
        epoch = self.epoch

        # estimate time
        if (cur_iter % (self.cfg.total_iter // 100) == 1 or cur_iter in [500, 1000]) and self.cfg.rank == 0:
            cost_t = (time.time() - self.s_time) / 60. / 60.
            self.s_time = time.time()  # recount
            self.pre_used_time += cost_t  # accumulate cost time

            left_t = self.pre_used_time / (self.cur_iter) * (self.cfg.total_iter - self.cur_iter)
            total_t = self.pre_used_time + left_t
            print(f'\nEpoch {epoch:.0f} cost {self.pre_used_time:.2f}h, total {total_t:.1f}h, left {left_t:.1f}h')

        # update lr
        self.scheduler.step()

    def print_iter_info(self, loss, psnr, pinfo=''):
        # self.epoch = -1
        cur_iter = self.cur_iter

        # torch.distributed.reduce(loss, dst=0)
        if self.cfg.rank == 0:
            # loss /= self.cfg.world_size
            self.lossess.append(loss.item())

            if cur_iter % self.cfg.print_freq == 0 or cur_iter == self.cfg.total_iter:
                print(f'\rTrain: %d %d/%d psnr=%.2f loss=%.4f ave_loss=%.4f %s' % (
                    self.epoch, cur_iter, self.cfg.total_iter, psnr, loss.item(), np.mean(self.lossess), pinfo),
                      self.cfg.name, '%.1e' % self.optimizer.param_groups[0]['lr'], end='')
                # if cur_iter % (self.cfg.print_freq * 20) == 0:
                #     self.writer.add_scalar('loss', loss.item(), cur_iter)

            if (cur_iter % self.cfg.save_img_freq == 1) or cur_iter in \
                    [self.cfg.total_iter] + [5 * (2 ** i) for i in range(9)]:
                if self.cfg.zero_mean:
                    save_img(self.cfg.result_dir, 'epoch_%d_s%d_%s.jpg' % (self.epoch, cur_iter, self.cfg.name),
                             self.input + 0.5, self.output + 0.5, self.gt + 0.5, rgb=True)
                else:
                    save_img(self.cfg.result_dir, 'epoch_%d_s%d_%s.jpg' % (self.epoch, cur_iter, self.cfg.name),
                             self.input, self.output, self.gt, rgb=True)

            if (cur_iter % self.cfg.save_state_freq == 1) or cur_iter == self.cfg.total_iter:
                self.save_training_state()

            # save model
            if (cur_iter % self.cfg.save_freq == 0) or cur_iter == self.cfg.total_iter:
                self.test()

            if cur_iter == (self.cfg.total_iter - 1) or cur_iter % self.cfg.save_freq == 0:
                # model_name = '%d.%d.psnr%s_%2f.pth' % (self.epoch, cur_iter, self.cfg.name, 0)
                # torch.save(self.net.module.state_dict(), os.path.join(self.cfg.model_dir, model_name))
                # if hasattr(self, 'net_ema'):
                #     model_name = '%d.%d.psnr%s_%2f.pth' % (self.epoch, cur_iter + 1, self.cfg.name, 0)
                #     torch.save(self.net_ema.state_dict(), os.path.join(self.cfg.model_dir, model_name))

                psnr = self.test_psnr if hasattr(self, 'test_psnr') else 0
                self.save_model_dict(psnr)

    def save_model_dict(self, psnr=0):
        model_name = '%d.%d.psnr%s_%2f.pth' % (self.epoch, self.cur_iter, self.cfg.name, psnr)
        torch.save(self.net.module.state_dict(), os.path.join(self.cfg.model_dir, model_name))
        if hasattr(self, 'net_ema') and self.cur_iter - 1 > 0:
            model_name = '%d.%d.ema.psnr%s_%2f.pth' % (self.epoch, self.cur_iter - 1, self.cfg.name, psnr)
            torch.save(self.net_ema.state_dict(), os.path.join(self.cfg.model_dir, model_name))

    def save_training_state(self):
        state = {'iter': self.cur_iter, 'optimizer': self.optimizer.state_dict(), 'net': self.net.state_dict(),
                 'pre_used_time': self.pre_used_time, 'epoch': self.epoch}
        retry = 3
        while retry > 0:
            try:
                torch.save(state, os.path.join(self.cfg.model_dir, 'state.pth'))
            except Exception as e:
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1
        if retry == 0:  # still cannot save
            raise IOError(f'Cannot save state model.')

    def resume_training_state(self):
        state_path = os.path.join(self.cfg.model_dir, 'state.pth')
        if not os.path.exists(state_path):
            if self.cfg.rank == 0: print('** NOT FIND ' + state_path)
        else:
            state = torch.load(state_path, map_location=lambda storage, loc: storage)
            self.cur_iter = state.get('iter', 0)
            self.pre_used_time = state.get('pre_used_time', 0)
            self.epoch = state.get('epoch', 0)

            self.optimizer.load_state_dict(state['optimizer'])
            self.net.load_state_dict(state['net'])

            if self.cfg.rank == 0: print('** resume the training', self.cur_iter)


@TRAINER_REGISTRY.register()
class DenoiseBase(Base_trainer):
    def __init__(self, cfg, train=True):
        super(DenoiseBase, self).__init__(cfg, train)

    def preprocess(self, data):
        if isinstance(data, dict):
            self.gt = data['gt']
        else:
            self.gt = data[1]

        self.gt = self.gt.permute(0, 3, 1, 2)
        self.input, _, _ = add_noise(self.gt, self.cfg)
        # print('*** clip')
        # self.input = self.input.clamp(0., 255.)
        # print('rank', self.cfg.rank, 'task id', self.task_id)


@TRAINER_REGISTRY.register()
class DenoiseBase_fastIDR(Base_trainer):
    def __init__(self, cfg, train=True):
        super(DenoiseBase_fastIDR, self).__init__(cfg, train)
        # self.net = self.model_to_device(build_model(cfg)).train()

        self.net_copy = self.model_to_device(build_model(cfg)).eval()
        self.net_copy_epoch = self.epoch

        # self.set_optimizer(cfg)
        # self.set_scheduler(cfg)

    # def model_to_device(self, net):
    #     net = net.cuda()  # move to cuda first, then call distributed data parallel.
    #     net = torch.nn.DataParallel(net)
    #     return net

    def preprocess(self, data):
        if isinstance(data, dict):
            clean = data['gt']
        else:
            clean = data[1]

        clean = clean.permute(0, 3, 1, 2)
        noisy, _, _ = add_noise(clean, self.cfg)

        if self.cfg.zero_mean:
            clean, noisy = clean - 0.5, noisy - 0.5

        self.gt = noisy
        # Get online cleaner targets
        if self.epoch >= 1:
            if self.net_copy_epoch < self.epoch:
                if self.cfg.rank == 0:
                    print('update parameters')
                self.net_copy_epoch = self.epoch
                self.net_copy.load_state_dict(self.net.state_dict())

            with torch.no_grad():
                self.gt = self.net_copy(noisy)

        self.input, _, _ = add_noise(self.gt, self.cfg)


@TRAINER_REGISTRY.register()
class DenoiseBase_n2n(Base_trainer):
    def __init__(self, cfg, train=True):
        super(DenoiseBase_n2n, self).__init__(cfg, train)

    def preprocess(self, data):
        if isinstance(data, dict):
            clean = data['gt']
        else:
            clean = data[1]

        clean = clean.permute(0, 3, 1, 2)
        noisy, _, _ = add_noise(clean, self.cfg)

        if self.cfg.zero_mean:
            clean, noisy = clean - 0.5, noisy - 0.5

        self.input, _, _ = add_noise(clean, self.cfg)
        self.gt = noisy


@TRAINER_REGISTRY.register()
class DenoiseBase_n2c(Base_trainer):
    def __init__(self, cfg, train=True):
        super(DenoiseBase_n2c, self).__init__(cfg, train)

    def preprocess(self, data):
        if isinstance(data, dict):
            clean = data['gt']
        else:
            clean = data[1]

        clean = clean.permute(0, 3, 1, 2)
        noisy, _, _ = add_noise(clean, self.cfg)

        if self.cfg.zero_mean:
            clean, noisy = clean - 0.5, noisy - 0.5

        self.input = noisy
        self.gt = clean


# todo: training mask for n2v and n2s
@TRAINER_REGISTRY.register()
class DenoiseBase_mask(Base_trainer):
    def __init__(self, cfg, train=True):
        super(DenoiseBase_mask, self).__init__(cfg, train)

    def preprocess(self, data):
        if isinstance(data, dict):
            clean = data['gt']
        else:
            clean = data[1]

        clean = clean.permute(0, 3, 1, 2)
        noisy, _, _ = add_noise(clean, self.cfg)

        if self.cfg.zero_mean:
            clean, noisy = clean - 0.5, noisy - 0.5

        self.input, _, _ = add_noise(clean, self.cfg)
        self.gt = noisy

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.output = self.net(self.input)

        loss = basic_loss(self.output, self.gt, type=self.cfg.loss_type)

        loss.backward()

        # psnr = 0  # get_criterion(loss)
        if self.cfg.clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.clip_norm)

        self.optimizer.step()
        self.print_iter_info(loss, 0)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        return loss
