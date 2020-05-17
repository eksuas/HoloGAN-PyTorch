import argparse
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
    parser.add_argument('--d-lr',           type=float,           default=0.0001)
    parser.add_argument('--g-lr',           type=float,           default=0.0001)
    parser.add_argument('--beta1',          type=float,           default=0.5)
    parser.add_argument('--beta2',          type=float,           default=0.999)
    parser.add_argument('--DStyle-lambda',  type=float,           default=1.0)
    parser.add_argument('--lambda-latent',  type=float,           default=0.0)
    parser.add_argument('--elevation-low',  type=int,             default=70)
    parser.add_argument('--elevation-high', type=int,             default=110)
    parser.add_argument('--azimuth-low',    type=int,             default=220)
    parser.add_argument('--azimuth-high',   type=int,             default=320)
    parser.add_argument('--scale-low',      type=float,           default=1.0)
    parser.add_argument('--scale-high',     type=float,           default=1.0)
    parser.add_argument('--transX-low',     type=int,             default=0)
    parser.add_argument('--transX-high',    type=int,             default=0)
    parser.add_argument('--transY-low',     type=int,             default=0)
    parser.add_argument('--transY-high',    type=int,             default=0)
    parser.add_argument('--transZ-low',     type=int,             default=0)
    parser.add_argument('--transZ-high',    type=int,             default=0)
    parser.add_argument('--no-save-model',  action='store_true',  default=False,
                                            help='do not save the current model')
    parser.add_argument('--load-dis',       type=str,             default=None,     metavar='S',
                                            help='the path for loading and evaluating discriminator')
    parser.add_argument('--load-gen',       type=str,             default=None,     metavar='S',
                                            help='the path for loading and evaluating generator')
    parser.add_argument('--device',         help=argparse.SUPPRESS)
    parser.add_argument('--start-epoch',    help=argparse.SUPPRESS)
    parser.add_argument('--recorder',       help=argparse.SUPPRESS)
    parser.add_argument('--results-dir',    help=argparse.SUPPRESS)
    parser.add_argument('--models-dir',     help=argparse.SUPPRESS)
    parser.add_argument('--hist-file',      help=argparse.SUPPRESS)
    #pylint: enable=C0326, C0330
    return parser.parse_args()

def main():
    args = initializer()
    model = HoloGAN(args)
    model.train(args)

if __name__ == '__main__':
    main()
