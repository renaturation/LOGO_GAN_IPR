import math
import time

import numpy as np
import torch

from pytorch_msssim import ssim as ssim_fn
from torch.nn import functional as F

from tqdm import tqdm

from dcgan import DCGAN
from watermark_fg import TransformVar, PasteWatermark
from inception import InceptionActivations
from loss import ssim  # 额外的损失
from util import DisableBatchNormStats, calculate_frechet_distance, calculate_inception_score
from util import compute_matching_prob  # 计算水印匹配度


class WatermarkDCGAN(DCGAN):
    def __init__(self, bbox=True, ds='CIFAR10', wm_img='A', num_wm=1):
        super(WatermarkDCGAN, self).__init__(ds)

        self.bbox = bbox

        # bbox
        if bbox:
            self.exp_name = 'DCGAN-%s-%s-%d' % (self.ds.upper(), wm_img, num_wm)
            self.Lambda = 1.0

            self.wm_file = './data/watermarks/%s.png' % wm_img
            self.wm_size = self.image_size // 2

            self.fn_inp = TransformVar(device=self.device, wm_img=wm_img, num1=num_wm, z_dim=self.z_dim).to(self.device)
            self.fn_out = PasteWatermark(watermark_size=self.wm_size, watermark_file=self.wm_file).to(self.device)

            self.xwm, self.ywm, self.Gxwm = None, None, None

            # loss
            self.loss_fn = ssim(normalized=True)
            self.LossG_bbox, self.LossW = torch.Tensor(0), torch.Tensor(0)

            # EVALUATE
            self.metrics_path = self.log_path + '%s.json' % self.exp_name
            self.evaluation_p_threshold = 0.01

            print('*** BLACK-BOX ***')
            # self.fn_inp = TransformVar(device=self.device, num1=f_num1, z_dim=self.z_dim).to(self.device)
            # self.fn_out = PasteWatermark(watermark_file=self.bbox_config['watermark_file']).to(self.device)

            print('Input f(x): TransformVar')
            print('Output g(x): PasteWatermark')
            print('lambda: %.4f' % self.Lambda)
            print('Loss: ssim')
            print('Experimental Name: %s' % self.exp_name)
            print()

    def get_metrics(self):
        if self.bbox:
            return {
                'D/Sum': self.LossD.item(),
                'D/Real': self.LossR.item(),
                'D/Fake': self.LossF.item(),
                'G/Sum': self.LossG_bbox.item() + self.Lambda * self.LossW.item(),  # Wrappers get_metrics
                'G/Adv': self.LossA.item(),
                'P/ssim': self.LossW.item()  # Wrappers get_metrics
            }
        else:
            return super().get_metrics()

    def model_train(self, d_iter=1, g_iter=1):
        # fetch data
        for _ in range(d_iter):
            x, _ = next(self.data_loader)
            z = torch.randn(x.size(0), self.z_dim)

            self.latent = z.to(self.device)
            self.real_sample = x.to(self.device)
            self.fake_sample = self.G(self.latent)

            real_out = self.D(self.real_sample)
            fake_out = self.D(self.fake_sample.detach())

            # hinge loss
            self.LossR = F.relu(1. - real_out).mean()
            self.LossF = F.relu(1. + fake_out).mean()
            self.LossD = self.LossR + self.LossF

            self.optD.zero_grad()
            self.LossD.backward()
            self.optD.step()

        for _ in range(g_iter):
            self.generated = self.fake_sample
            fake_out = self.D(self.generated)

            # adversarial loss
            self.LossA = - fake_out.mean()
            self.LossG = self.LossA

            if self.bbox:
                # Wrappers forward_g
                x = self.latent
                y = self.generated

                with torch.no_grad():
                    self.xwm = self.fn_inp(x.detach())
                    self.ywm = self.fn_out(y.detach())

                G = self.G
                with DisableBatchNormStats(G):
                    self.Gxwm = G(self.xwm)

                # Wrappers compute_g_loss
                self.LossG_bbox = self.LossG
                self.LossW = self.loss_fn(self.Gxwm, self.ywm)

                Loss = self.LossG_bbox + self.Lambda * self.LossW
            else:
                Loss = self.LossG

            # Wrappers  update_g
            self.optG.zero_grad()
            Loss.backward()
            self.optG.step()

    def model_evaluate(self):
        assert self.bbox, "self.bbox is False, No need to evaluate."

        def post_proc(xx):
            return (xx.clamp_(-1, 1) + 1.) / 2.

        # apply_mask = self.fn_out.__class__(watermark_size=self.image_size//2, opaque=True, normalized=True).apply_mask
        apply_mask = self.fn_out.apply_mask

        torch.manual_seed(self.seed)

        print('*** EVALUATION ***')

        inception = InceptionActivations().to(self.device)

        self.G.eval()

        loader = self.data_loader

        stats = {'fx': [], 'fy': [], 'prob': [], 'q': [], 'p': [], 'm': []}

        sample_iter = 600
        sample_size = sample_iter * self.batch_size
        sample_tqdm = tqdm(range(sample_iter))
        sample_tqdm.set_description(desc=self.ds)

        for _ in sample_tqdm:
            y, a = next(loader)
            with torch.no_grad():
                z = torch.randn(y.size(0), 128).to(self.device)
                x = self.G(z)

                zwm = self.fn_inp(z)
                xwm = self.G(zwm)
                ywm = self.fn_out(x)

                # wm_x = post_proc(apply_mask(xwm.cpu()))
                # wm_y = post_proc(apply_mask(ywm.cpu()))
                wm_x = post_proc(apply_mask(xwm))
                wm_y = post_proc(apply_mask(ywm))

                ssim = ssim_fn(wm_x, wm_y, data_range=1, size_average=False)
                p_value = compute_matching_prob(wm_x, wm_y)
                match = p_value < self.evaluation_p_threshold

                stats['q'].append(ssim.detach().cpu())
                stats['p'].append(p_value)
                stats['m'].append(match)

                fx, prob = inception(x.detach())
                fy, _ = inception(y.to(self.device))
                stats['fx'].append(fx.detach().cpu())
                stats['fy'].append(fy.detach().cpu())
                stats['prob'].append(prob.detach().cpu())

        for k in stats:
            stats[k] = torch.cat(stats[k], dim=0).numpy()

        mu1 = np.mean(stats['fx'], axis=0)
        mu2 = np.mean(stats['fy'], axis=0)
        sig1 = np.cov(stats['fx'], rowvar=False)
        sig2 = np.cov(stats['fy'], rowvar=False)

        fid = calculate_frechet_distance(mu1, sig1, mu2, sig2)
        is_mean, is_std = calculate_inception_score(stats['prob'])
        ssim_wm = np.mean(stats['q'])
        p_value = np.mean(stats['p'])
        match = np.sum(stats['m'])

        # metrics = {}

        # metrics[self.ds] = {
        #     'FID': f'{fid:.4f}',
        #     'IS_MEAN': f'{is_mean:.4f}',
        #     'IS_STD': f'{is_std:.4f}'
        # }
        #
        # metrics[self.ds]['BBOX'] = {
        #     'Q_WM': f'{ssim_wm:.4f}',
        #     'P': f'{p_value:.3e}',
        #     'MATCH': f'{match:d}/{sample_size:d}'
        # }

        print(
            f'\nDataset: {self.ds}'
            f'\n\tFID: {fid:.4f}'
            f'\n\tIS: {is_mean:.4f} +/- {is_std:.4f}'
            f'\n\tBBOX:'
            f'\n\t\tQ_WM: {ssim_wm:.4f}'
            f'\n\t\tP: {p_value:.3e}'
            f'\n\t\tMATCH: {match / sample_size:.4f}'
            f'\n'
        )

        # json.dump(metrics, open(self.metrics_path, 'w'), indent=2, sort_keys=True)
        # print(f'Result saved to: {self.metrics_path}')
        import csv
        import os
        import shutil
        with open('log/train.csv', 'a+', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            # writer.writerow(["Experiment", "FID", "IS平均值", "IS标准差", "水印质量", "P值", "匹配度"])
            writer.writerow([self.exp_name, f'{fid:.4f}', f'{is_mean:.4f}', f'{is_std:.4f}', f'{ssim_wm:.4f}',
                             f'{p_value:.3e}', match, f'{match / sample_size:.4f}'])

        save_path = './log/%s/' % self.exp_name
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_list = os.listdir('./log/')
        for f in file_list:
            if str(f).startswith('check') or str(f).startswith('events'):
                shutil.move('./log/' + f, save_path + f)
                # print(f)

    def bbox_security_analysis(self, change=0):
        assert self.bbox, "self.bbox is False, No need to evaluate."

        def post_proc(xx):
            return (xx.clamp_(-1, 1) + 1.) / 2.

        apply_mask = self.fn_out.apply_mask

        torch.manual_seed(self.seed)

        print('*** SECURITY ANALYSIS ***')
        time.sleep(1)
        self.G.eval()

        loader = self.data_loader

        sample_iter = 500
        sample_size = sample_iter * self.batch_size
        sample_tqdm = tqdm(range(sample_iter))
        sample_tqdm.set_description(desc=self.ds)

        ssim1, ssim2, ssim3 = [], [], []
        match1, match2, match3 = 0, 0, 0
        for _ in sample_tqdm:
            y, a = next(loader)
            with torch.no_grad():
                z = torch.randn(y.size(0), 128).to(self.device)  # z
                x = self.G(z)  # y

                zwm = self.fn_inp(z)  # z_wm
                xwm = self.G(zwm)  # y_wm
                ywm = self.fn_out(x)  # g(y)

                adv_fn_inp = TransformVar(device=self.device, wm_img='A', num1=35, change=change).to(self.device)
                adv_zwm = adv_fn_inp(z)
                adv_xwm = self.G(adv_zwm)

                x = post_proc(x[..., 0:self.image_size//2, 0:self.image_size//2])
                wm_x = post_proc(apply_mask(xwm))
                wm_y = post_proc(apply_mask(ywm))
                adv_wm_x = post_proc(apply_mask(adv_xwm))

                s1 = ssim_fn(wm_x, wm_y, data_range=1, size_average=False)
                ssim1.append(torch.mean(s1).item())
                p_value1 = compute_matching_prob(wm_x, wm_y)
                m1 = p_value1 < self.evaluation_p_threshold
                match1 += torch.sum(m1).item()

                s2 = ssim_fn(x, wm_y, data_range=1, size_average=False)
                ssim2.append(torch.mean(s2).item())
                p_value2 = compute_matching_prob(x, wm_y)
                m2 = p_value2 < self.evaluation_p_threshold
                match2 += torch.sum(m2).item()

                s3 = ssim_fn(adv_wm_x, wm_y, data_range=1, size_average=False)
                ssim3.append(torch.mean(s3).item())
                p_value3 = compute_matching_prob(adv_wm_x, wm_y)
                m3 = p_value3 < self.evaluation_p_threshold
                match3 += torch.sum(m3).item()

        ssim1 = np.array(ssim1)
        ssim2 = np.array(ssim2)
        ssim3 = np.array(ssim3)

        import csv
        with open('./log/security.csv', 'a+', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            if change == 0:
                writer.writerow([f'{self.exp_name}-SSIM(g(y), y_wm)', f'{np.max(ssim1):.4f}', f'{np.mean(ssim1):.4f}',
                                 f'{np.min(ssim1):.4f}', f'{match1:.0f}', f'{match1 / sample_size:.4f}']
                                + ["%.8f" % s for s in ssim1])
                writer.writerow([f'{self.exp_name}-SSIM(y, y_wm)', f'{np.max(ssim2):.4f}', f'{np.mean(ssim2):.4f}',
                                 f'{np.min(ssim2):.4f}', f'{match2:.0f}', f'{match2 / sample_size:.4f}']
                                + ["%.8f" % s for s in ssim2])
            writer.writerow([f'{self.exp_name}-SSIM(adv_g(y), y_wm)(HD={change})', f'{np.max(ssim3):.4f}', f'{np.mean(ssim3):.4f}',
                             f'{np.min(ssim3):.4f}', f'{match3:.0f}', f'{match3 / sample_size:.4f}']
                            + ["%.8f" % s for s in ssim3])


