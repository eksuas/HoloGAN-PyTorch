"""
HoloGAN implementation in PyTorch
May 17, 2020
"""
import os
import csv
import time
import math
import collections
import torch
import numpy as np
#import matplotlib.pyplot as plt

from torch import nn
from torch.optim import Adam
from torch.autograd import Variable
from torchvision import datasets, transforms
from scipy.misc import imsave
from discriminator import Discriminator
from generator import Generator

class HoloGAN():
    """HoloGAN.

    HoloGAN model is the Unsupervised learning of 3D representations from natural images.
    The paper can be found in https://www.monkeyoverflow.com/hologan-unsupervised-learning-\
    of-3d-representations-from-natural-images/
    """
    def __init__(self, args):
        super(HoloGAN, self).__init__()

        torch.manual_seed(args.seed)
        use_cuda = args.gpu and torch.cuda.is_available()
        args.device = torch.device("cuda" if use_cuda else "cpu")

        # model configurations
        if args.load_dis is None:
            self.discriminator = Discriminator(in_planes=3, out_planes=64,
                                               z_planes=args.z_dim).to(args.device)
        else:
            self.discriminator = torch.load(args.load_dis).to(args.device)

        if args.load_gen is None:
            self.generator = Generator(in_planes=64, out_planes=3,
                                       z_planes=args.z_dim).to(args.device)
        else:
            self.generator = torch.load(args.load_gen).to(args.device)

        # optimizer configurations
        self.optimizer_discriminator = Adam(self.discriminator.parameters(),
                                            lr=args.d_lr, betas=(args.beta1, args.beta2))
        self.optimizer_generator = Adam(self.generator.parameters(),
                                        lr=args.d_lr, betas=(args.beta1, args.beta2))

        # Load dataset
        self.train_loader = self.load_dataset(args)

        # create result folder
        args.results_dir = "./results/"+args.dataset
        if not os.path.exists(args.results_dir):
            os.makedirs(args.results_dir)

        # create history file
        args.hist_file = open(args.results_dir+"/history.csv", "a", newline="")
        args.recorder = csv.writer(args.hist_file, delimiter=",")
        if os.stat(args.results_dir+"/history.csv").st_size == 0:
            args.recorder.writerow(["epoch", "time", "d_loss", "g_loss", "q_loss"])

        # create model folder
        args.models_dir = "./models/"+args.dataset
        if not os.path.exists(args.models_dir):
            os.makedirs(args.models_dir)

        # continue to broken training
        args.start_epoch = 0
        while os.path.exists(args.models_dir+"/discriminator.v"+str(args.start_epoch)+".pt") and \
              os.path.exists(args.models_dir+"/generator.v"+str(args.start_epoch)+".pt"):
            args.start_epoch += 1

        # create sampling folder
        args.samples_dir = "./samples/"+args.dataset
        if not os.path.exists(args.samples_dir):
            os.makedirs(args.samples_dir)

    def train(self, args):
        """HoloGAN trainer

        This method train the HoloGAN model.
        """
        for epoch in range(args.start_epoch, args.max_epochs):
            result = collections.OrderedDict({"epoch":epoch})
            print("Epoch: [{:2d}] ".format(epoch), end="")

            result.update(self.train_epoch(args))
            # validate and keep history at each log interval
            self.save_history(args, result)

            # save model parameters
            if not args.no_save_model:
                self.save_model(args, epoch)

        # save the model giving the best validation results as a final model
        if not args.no_save_model:
            self.save_model(args, args.max_epochs-1, True)

    def train_epoch(self, args):
        """train an epoch

        This method train an epoch.
        """
        batch = {"time":[], "g":[], "d":[], "q":[]}
        self.generator.train()
        self.discriminator.train()
        for idx, (data, _) in enumerate(self.train_loader):
            print("[{:3d}/{:3d}] ".format(idx, len(self.train_loader)), end="")
            x = data.to(args.device)
            # rnd_state = np.random.RandomState(seed)
            z = self.sample_z(args)
            view_in = self.sample_view(args)

            d_loss, g_loss, q_loss, elapsed_time = self.train_batch(x, z, view_in, args)
            batch["d"].append(float(d_loss))
            batch["g"].append(float(g_loss))
            batch["q"].append(float(q_loss))
            batch["time"].append(float(elapsed_time))

            # print the training results of batch
            print("time: {:.2f}sec, d_loss: {:.4f}, g_loss: {:.4f}, q_loss: {:.4f}"
                  .format(elapsed_time, float(d_loss), float(g_loss), float(q_loss)))

        result = {"time"  : round(np.mean(batch["time"])),
                  "d_loss": round(np.mean(batch["d"]), 4),
                  "g_loss": round(np.mean(batch["g"]), 4),
                  "q_loss": round(np.mean(batch["q"]), 4)}
        return result

    def train_batch(self, x, z, view_in, args):
        """train the given batch

        Arguments are
        * x:        images in the batch.
        * z:        latent variables in the batch.
        * view_in:  3D transformation parameters.

        This method train the given batch and return the resulting loss values.
        """
        start = time.process_time()
        loss = nn.BCEWithLogitsLoss()
        self.optimizer_generator.zero_grad()
        fake = self.generator(z, view_in)
        d_fake, g_z_pred, g_t_pred = self.discriminator(fake[:, :, :64, :64])
        gen_loss = loss(d_fake, torch.ones(d_fake.shape))
        g_z_loss = torch.mean((g_z_pred-z)**2)
        g_t_loss = torch.mean((g_t_pred-z)**2)

        # if (kwargs['iter']-1) % self.update_g_every == 0:
        (gen_loss + args.lambda_latent * (g_z_loss + g_t_loss)).backward()
        self.optimizer_generator.step()

        self.optimizer_discriminator.zero_grad()
        d_fake, d_z_pred, d_t_pred = self.discriminator(fake[:, :, :64, :64].detach())
        d_real, _, _ = self.discriminator(x)
        dis_loss = loss(d_real, torch.ones(d_real.shape)) + loss(d_fake, torch.zeros(d_fake.shape))
        d_z_loss = torch.mean((d_z_pred-z)**2)
        d_t_loss = torch.mean((d_t_pred-z)**2)
        (dis_loss + args.lambda_latent * (d_z_loss + d_t_loss)).backward()
        self.optimizer_discriminator.step()

        elapsed_time = time.process_time()  - start
        return float(dis_loss), float(gen_loss), float(g_z_loss + g_t_loss), elapsed_time

    def sample(self, args):
        """HoloGAN sampler

        This samples images in the given configuration from the HoloGAN.
        Images can be found in the "args.samples_dir" directory.
        """
        z = self.sample_z(args)
        if args.rotate_azimuth:
            low, high, step = args.azimuth_low, args.azimuth_high, 10
        elif args.rotate_elevation:
            low, high, step = args.elevation_low, args.elevation_high, 5
        else:
            low, high, step = 0, 10, 1

        for i in range(low, high, step):
            # Apply only azimuth rotation
            if args.rotate_azimuth:
                view_in = torch.tensor([i*math.pi/180, 0, 1.0, 0, 0, 0])
                view_in = view_in.repeat(args.batch_size, 1)
            # Apply only elevation rotation
            elif args.rotate_elevation:
                view_in = torch.tensor([270*math.pi/180, (90-i)*math.pi/180, 1.0, 0, 0, 0])
                view_in = view_in.repeat(args.batch_size, 1)
            # Apply default transformation
            else:
                view_in = self.sample_view(args)

            samples = self.generator(z, view_in).permute(0, 2, 3, 1)
            normalized = ((samples+1.)/2.).detach().numpy()
            image = np.clip(255*normalized, 0, 255).astype(np.uint8)
            imsave(os.path.join(args.samples_dir, "samples_{}.png".format(i)), image[0])

    def load_dataset(self, args):
        """dataset loader.

        This loads the dataset.
        """
        kwargs = {'num_workers': 2, 'pin_memory': True} if args.device == 'cuda' else {}

        if args.dataset == 'celebA':
            transform = transforms.Compose([\
                transforms.CenterCrop(108),
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

            trainset = datasets.ImageFolder(root=args.image_path, transform=transform)
            #trainset = datasets.ImageFolder(root=root+'/train', transform=transform)
            #testset = datasets.ImageFolder(root=root+'/val', transform=transform)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,\
                        shuffle=True, **kwargs)
        return train_loader

    def sample_z(self, args):
        """Latent variables sampler

        This samples latent variables from the uniform distribution [-1,1].
        """
        tensor = torch.cuda.FloatTensor if args.device == "cuda" else torch.FloatTensor
        size = (args.batch_size, args.z_dim)
        return Variable(tensor(np.random.uniform(-1., 1., size)), requires_grad=True)

    def sample_view(self, args):
        """Transformation parameters sampler

        This samples view (or transformation parameters) from the given configuration.
        """
        # the azimuth angle (theta) is around y
        theta = np.random.randint(args.azimuth_low, args.azimuth_high,
                                  (args.batch_size)).astype(np.float)
        theta = (theta - 90.) * math.pi / 180.0
        # the elevation angle (gamma) is around x
        gamma = np.random.randint(args.elevation_low, args.elevation_high,
                                  (args.batch_size)).astype(np.float)
        gamma = (90. - gamma) * math.pi / 180.0
        scale = float(np.random.uniform(args.scale_low, args.scale_high))
        shift_x = args.transX_low + np.random.random(args.batch_size) * \
                  (args.transX_high - args.transX_low)
        shift_y = args.transY_low + np.random.random(args.batch_size) * \
                  (args.transY_high - args.transY_low)
        shift_z = args.transZ_low + np.random.random(args.batch_size) * \
                  (args.transZ_high - args.transZ_low)

        view = np.zeros((args.batch_size, 6))
        column = np.arange(0, args.batch_size)
        view[column, 0] = theta
        view[column, 1] = gamma
        view[column, 2] = scale
        view[column, 3] = shift_x
        view[column, 4] = shift_y
        view[column, 5] = shift_z
        return view

    def save_history(self, args, record):
        """save a record to the history file"""
        args.recorder.writerow([str(record[key]) for key in record])
        args.hist_file.flush()

    def save_model(self, args, epoch, best=False):
        """save model

        Arguments are
        * epoch:   epoch number.
        * best:    if the model is in the final epoch.

        This method saves the trained discriminator and generator in a pt file.
        """
        filename = args.models_dir
        if best is False:
            torch.save(self.discriminator, filename+"/discriminator.v"+str(epoch)+".pt")
            torch.save(self.generator, filename+"/generator.v"+str(epoch)+".pt")
        else:
            train_files = os.listdir(args.models_dir)
            for train_file in train_files:
                if not train_file.endswith(".v"+str(epoch)+".pt"):
                    os.remove(os.path.join(args.models_dir, train_file))
            os.rename(filename+"/discriminator.v"+str(epoch)+".pt", filename+"/discriminator.pt")
            os.rename(filename+"/generator.v"+str(epoch)+".pt", filename+"/generator.pt")
