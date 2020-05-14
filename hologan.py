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
                                            lr=args.d_eta, betas=(args.beta1, args.beta2))
        self.optimizer_generator = Adam(self.generator.parameters(),
                                        lr=args.d_eta, betas=(args.beta1, args.beta2))

        # TODO: create result folder

        # TODO: create model folder

        # TODO: continue to broken training

    def train(self, args):
        train_loader = self.load_dataset(args)
        tensor = torch.cuda.FloatTensor if args.device == "cuda" else torch.FloatTensor
        self.view_in = 6

        for epoch in range(1, args.max_epochs + 1):
            print('{:3d}: '.format(epoch), end='')
            losses = []
            self.generator.train()
            self.discriminator.train()
            for data, _ in train_loader:
                data = data.to(args.device)
                #optimizer.zero_grad()

                z = Variable(tensor(np.random.normal(size=(args.batch_size, args.z_dim))))
                print("generator \n", self.generator(z).shape)
                print("discriminator \n", self.discriminator(data))

                #loss = F.cross_entropy(pred, target)

                #losses.append(float(loss))
                #loss.backward()
                #optimizer.step()

        plt.show()

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
