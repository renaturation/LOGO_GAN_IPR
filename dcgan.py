# copy from models

from torch import optim
from torch.nn.utils import spectral_norm as SN
import os
import time

from torch import nn
import torch
from torch.nn import functional as F
from tqdm import trange

from datasets import cifar10, celeba
from util import Logger


# ---------------------- Generator ---------------------- #
# copy from networks
def block_g(n_inp: int, n_out: int):
    r = nn.Sequential(
        nn.ConvTranspose2d(n_inp, n_out, 4, 2, 1, bias=False),
        nn.BatchNorm2d(n_out),
        nn.ReLU(inplace=True)
    )

    return r


class ConvGenerator(nn.Module):
    def __init__(self, mg=4, z_dim=128):
        super(ConvGenerator, self).__init__()

        self.mg = mg
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 512 * mg * mg),
            nn.ReLU(inplace=True)
        )
        self.convs = nn.Sequential(
            block_g(512, 256),
            block_g(256, 128),
            block_g(128, 64),
            nn.ConvTranspose2d(64, 3, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        z = self.fc(z).view(z.size(0), -1, self.mg, self.mg)
        return self.convs(z)


# ---------------------- Discriminator ---------------------- #
# copy from networks
def block_d(n_inp: int, n_out: int):
    r = nn.Sequential(
        SN(nn.Conv2d(n_inp, n_out, 3, 1, 1, bias=True)),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        SN(nn.Conv2d(n_out, n_out, 4, 2, 1, bias=True)),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
    )

    return r


class SNDiscriminator(nn.Module):
    def __init__(self, md=4):
        super(SNDiscriminator, self).__init__()

        self.md = md
        self.main = nn.Sequential(
            block_d(3, 64),
            block_d(64, 128),
            block_d(128, 256),
            SN(nn.Conv2d(256, 512, 3, 1, 1, bias=True)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.linear = SN(nn.Linear(512 * md * md, 1))

    def forward(self, x):
        x = self.main(x)
        x = x.view(-1, 512 * self.md * self.md)
        x = self.linear(x)
        return x.view(-1)


# ---------------------- DCGAN ---------------------- #
class DCGAN(nn.Module):
    def __init__(self, ds='CIFAR10'):
        super(DCGAN, self).__init__()

        # base parameters
        self.ds = ds
        self.image_size = 32
        self.batch_size = 50 if ds == 'CIFAR10' else 100

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = 1234

        # DATA
        self.data_loader = None

        # TRAIN
        # d loss  D/Real, D/Fake, D/Sum
        self.LossR, self.LossF, self.LossD = torch.Tensor(0), torch.Tensor(0), torch.Tensor(0)
        # g loss  G/Adv, G/Sum
        self.LossA, self.LossG = torch.Tensor(0), torch.Tensor(0)
        # data
        self.latent, self.real_sample, self.fake_sample, self.generated = None, None, None, None
        self.z_dim = 128

        self._step, self.init_step, self.end_step, self.max_step = 0, 0, 0, 100000
        self.log_path = './log/'
        self.log_freq = 1000
        self.logger = Logger(path=self.log_path)
        self.cp_path = self.log_path + 'checkpoint.pt'

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        # DCGAN
        self.G = ConvGenerator(mg=self.image_size//8, z_dim=self.z_dim).to(self.device)
        self.D = SNDiscriminator(md=self.image_size//8).to(self.device)

        self.lr = 2.0e-4
        self.optG = optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.optD = optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.5, 0.999))

        self.G.train()
        self.D.train()

        self.configure_dataset()
        self.configure_model()

    def configure_dataset(self, num_workers=4):
        print('*** DATASET ***')

        if self.ds == 'CIFAR10':
            assert self.image_size == 32, "The size of %s image is 32." % self.ds.upper()
            self.data_loader = cifar10(
                size=self.image_size,
                batch_size=self.batch_size,
                num_workers=num_workers,
                drop_last=True
            )
        elif self.ds in ['CelebA', 'CELEBA']:
            # assert self.image_size == 64, "The size of %s image is 64." % self.ds.upper()
            self.data_loader = celeba(
                size=self.image_size,
                batch_size=self.batch_size,
                num_workers=num_workers,
                drop_last=True
            )

        print(f'Name: {self.ds.upper()}')
        print(f'# samples: {len(self.data_loader)}\n')

    def configure_model(self):
        params_g = sum(map(lambda p: p.numel(), self.G.parameters()))
        params_d = sum(map(lambda p: p.numel(), self.D.parameters()))

        print('*** MODEL ***')
        print(f'Generator params: {params_g}')
        print(f'Discriminator params: {params_d}\n')

    def get_metrics(self):
        return {
            'D/Sum': self.LossD.item(),
            'D/Real': self.LossR.item(),
            'D/Fake': self.LossF.item(),
            'G/Sum': self.LossG.item(),
            'G/Adv': self.LossA.item()
        }

    def load_checkpoint(self):
        if os.path.exists(self.cp_path):
            print('*** LOAD CHECKPOINT ***')
            state_dict = torch.load(self.cp_path)

            self._step = state_dict['step']
            state_dict.pop('step')
            self.load_state_dict(state_dict)

            if self._step == self.max_step:
                self.init_step = self.max_step
            else:
                self.init_step = int(self._step) + 1
            self.end_step = self.max_step

            print(f'Train From Step: {self.init_step}\n')
        else:
            self.init_step = 0
            self.end_step = self.max_step

    def save_checkpoint(self):
        if self._step == self.max_step:
            state_dict = self.state_dict()
            state_dict['step'] = self.max_step
            torch.save(state_dict, self.log_path + 'checkpoint.pt')
            return

        metrics = self.get_metrics()
        self.logger.write_scalar(metrics, self._step)

        def post_proc(xx):
            return (xx.clamp_(-1, 1) + 1.) / 2.

        if (self._step + 1) % self.log_freq == 0:
            if not hasattr(self, 'fixed_z'):
                if self.bbox:
                    with torch.no_grad():
                        z = torch.randn(8 * 6, self.z_dim).to(self.device)
                        zwm = self.fn_inp(z).detach()
                        z = torch.cat([z, zwm], dim=0)
                else:
                    z = torch.randn(8 * 12, self.z_dim)
                self.fixed_z = z.to(self.device)

            with torch.no_grad():
                self.G.eval()
                fake_sample = self.G(self.fixed_z)
                self.G.train()
                img = post_proc(fake_sample).detach().cpu()

            self.logger.save_images(img, self._step)

            state_dict = self.state_dict()
            state_dict['step'] = self._step

            torch.save(state_dict, self.log_path + 'checkpoint.pt')

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

            self.optG.zero_grad()
            self.LossG.backward()
            self.optG.step()

    def model_evaluate(self):
        pass

    def start(self, train_flag=True, eval_flag=False):

        self.load_checkpoint()

        if train_flag:
            print('*** TRAIN ***')
            time.sleep(1)
            if self.init_step < self.end_step:
                for step in trange(self.init_step, self.end_step):
                    self._step = step
                    self.model_train()
                    self.save_checkpoint()
            else:
                print("预加载模型已训练完成")

            self._step = self.max_step
            self.save_checkpoint()
            print()

        if eval_flag:
            self.model_evaluate()
