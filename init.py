import argparse
import torch
from torch.optim import Adam
from hologan import HoloGAN

def initializer():
    """initializer of the program.

    This parses and extracts all training and testing settings.
    """
    #pylint: disable=C0326, C0330
    parser = argparse.ArgumentParser(description='PyTorch HoloGAN implementation')
    parser.add_argument('--seed',           type=int,             default=23)
    parser.add_argument('--image-path',     type=str,             default="../dataset/fake/celebA")
    parser.add_argument('--dataset',        type=str,             default="celebA", choices=["celebA"])
    parser.add_argument('--gpu',            action='store_true',  default=False)
    parser.add_argument('--batch-size',     type=int,             default=32)
    parser.add_argument('--max-epochs',     type=int,             default=50)
    parser.add_argument('--epoch-step',     type=int,             default=25)
    parser.add_argument('--z-dim',          type=int,             default=128)
    parser.add_argument('--d-eta',          type=float,           default=0.0001)
    parser.add_argument('--g-eta',          type=float,           default=0.0001)
    parser.add_argument('--beta1',          type=float,           default=0.5)
    parser.add_argument('--beta2',          type=float,           default=0.999)
    parser.add_argument('--DStyle-lambda',  type=float,           default=1.0)
    parser.add_argument('--lambda-latent',  type=float,           default=0.0)
    parser.add_argument('--ele-low',        type=int,             default=70)
    parser.add_argument('--ele-high',       type=int,             default=110)
    parser.add_argument('--azi-low',        type=int,             default=220)
    parser.add_argument('--azi-high',       type=int,             default=320)
    parser.add_argument('--scale-low',      type=float,           default=1.0)
    parser.add_argument('--scale-high',     type=float,           default=1.0)
    parser.add_argument('--x-low',          type=int,             default=0)
    parser.add_argument('--x-high',         type=int,             default=0)
    parser.add_argument('--y-low',          type=int,             default=0)
    parser.add_argument('--y-high',         type=int,             default=0)
    parser.add_argument('--z-low',          type=int,             default=0)
    parser.add_argument('--z-high',         type=int,             default=0)
    #pylint: enable=C0326, C0330
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    use_cuda = args.gpu and torch.cuda.is_available()
    args.device = torch.device('cuda' if use_cuda else 'cpu')

    # model configurations
    model = HoloGAN(**vars(args))

    # optimizer configurations
    optimizer = Adam(model.parameters(), lr=args.d_eta, betas=(args.beta1, args.beta2))

    # TODO: create result folder

    # TODO: create model folder

    # TODO: continue to broken training

    return model, optimizer, args
    """
    # Remaining configurations
    "style_disc":"false",
    "sample_z":"uniform",
    "add_D_noise":"false",
    "with_translation":"false",
    "with_scale":"false",
    "output_dir": "./HoloGAN"
     """
