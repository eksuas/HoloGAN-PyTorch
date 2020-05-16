import math
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.optim import Adam
from torch.autograd import Variable
from torchvision import datasets, transforms

from discriminator import Discriminator
from generator import Generator

class HoloGAN():
    def __init__(self, args):
        super(HoloGAN, self).__init__()

        torch.manual_seed(args.seed)
        use_cuda = args.gpu and torch.cuda.is_available()
        args.device = torch.device('cuda' if use_cuda else 'cpu')

        # model configurations
        self.discriminator = Discriminator(in_planes=3,  out_planes=64, z_planes=args.z_dim)
        self.generator     = Generator    (in_planes=64, out_planes=3,  z_planes=args.z_dim)

        # optimizer configurations
        self.optimizer_discriminator = Adam(self.discriminator.parameters(),
                                            lr=args.d_lr, betas=(args.beta1, args.beta2))
        self.optimizer_generator = Adam(self.generator.parameters(),
                                        lr=args.d_lr, betas=(args.beta1, args.beta2))

        # TODO: create result folder

        # TODO: create model folder

        # TODO: continue to broken training

    def train(self, args):
        train_loader = self.load_dataset(args)
        for epoch in range(1, args.max_epochs + 1):
            print('{:3d}: '.format(epoch), end='')
            losses = []
            self.generator.train()
            self.discriminator.train()

            loss = nn.BCEWithLogitsLoss()
            for data, _ in train_loader:

                data = data.to(args.device)
                # rnd_state = np.random.RandomState(seed)
                z = self.sample_z(args)
                view_in = self.sample_view(args)

                lamb = 0.0
                self.optimizer_generator.zero_grad()
                fake = self.generator(z, view_in)
                d_fake, g_z_pred , g_t_pred = self.discriminator(fake[:,:,:64,:64])
                gen_loss = loss(d_fake, torch.ones(d_fake.shape))
                g_z_loss = torch.mean((g_z_pred-z)**2)
                g_t_loss = torch.mean((g_t_pred-z)**2)

                # if (kwargs['iter']-1) % self.update_g_every == 0:
                (gen_loss + lamb*(g_z_loss + g_t_loss)).backward()
                self.optimizer_generator.step()

                self.optimizer_discriminator.zero_grad()
                d_fake, d_z_pred, d_t_pred = self.discriminator(fake[:,:,:64,:64].detach())
                d_real, _, _ = self.discriminator(data)
                d_loss = loss(d_real, torch.ones(d_real.shape)) + loss(d_fake, torch.zeros(d_fake.shape))
                d_z_loss = torch.mean((d_z_pred-z)**2)
                d_t_loss = torch.mean((d_t_pred-z)**2)
                (d_loss + lamb*(d_z_loss + d_t_loss)).backward()
                self.optimizer_discriminator.step()

    def sample(self, args):
        return

    def load_dataset(self, args):
        """dataset loader.

        This loads the dataset.
        """
        kwargs = {'num_workers': 2, 'pin_memory': True} if args.device == 'cuda' else {}

        if args.dataset == 'celebA':
            root = '../dataset/fake/celebA'

            transform = transforms.Compose([\
                transforms.CenterCrop(108),
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

            trainset = datasets.ImageFolder(root=root, transform=transform)
            #trainset = datasets.ImageFolder(root=root+'/train', transform=transform)
            #testset = datasets.ImageFolder(root=root+'/val', transform=transform)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,\
                        shuffle=True, **kwargs)
        return train_loader

    def sample_z(self, args):
        tensor = torch.cuda.FloatTensor if args.device == "cuda" else torch.FloatTensor
        size = (args.batch_size, args.z_dim)
        return Variable(tensor(np.random.uniform(-1., 1., size)), requires_grad=True)

    def sample_view(self, args):
        # the azimuth angle (theta) is around y
        theta = np.random.randint(args.azimuth_low, args.azimuth_high, (args.batch_size)).astype(np.float)
        theta = (theta - 90.) * math.pi / 180.0
        # the elevation angle (gamma) is around x
        gamma = np.random.randint(args.elevation_low, args.elevation_high, (args.batch_size)).astype(np.float)
        gamma = (90. - gamma) * math.pi / 180.0
        scale = float(np.random.uniform(args.scale_low, args.scale_high))
        shift_x = args.transX_low + np.random.random(args.batch_size) * (args.transX_high - args.transX_low)
        shift_y = args.transY_low + np.random.random(args.batch_size) * (args.transY_high - args.transY_low)
        shift_z = args.transZ_low + np.random.random(args.batch_size) * (args.transZ_high - args.transZ_low)

        view = np.zeros((args.batch_size, 6))
        column = np.arange(0, args.batch_size)
        view[column, 0] = theta
        view[column, 1] = gamma
        view[column, 2] = scale
        view[column, 3] = shift_x
        view[column, 4] = shift_y
        view[column, 5] = shift_z
        return view
