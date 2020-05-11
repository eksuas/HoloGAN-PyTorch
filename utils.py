import os
import argparse
import torch
import numpy as np
from PIL import Image
from torch.optim import Adam
from torch.optim import SGD

def initializer():
    """initializer of the program.

    This parses and extracts all training and testing settings.
    """
    #pylint: disable=C0326, C0330
    parser = argparse.ArgumentParser(description='PyTorch HoloGAN implementation')
    parser.add_argument('--seed',           type=int,             default=23)
    parser.add_argument('--image-path',     type=str,             default="./celebA")
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

    # TODO: model configurations

    # TODO: optimizer configurations

    # TODO: create result folder

    # TODO: create model folder

    # TODO: continue to broken training

    # TODO: return model, optimizer, args

    return args
    """
    # Remaining configurations
    "discriminator":"discriminator_IN",
    "generator":"generator_AdaIN8",
    "view_func":"generate_random_rotation_translation",
    "train_func":"train_HoloGAN",
    "build_func":"build_HoloGAN",
    "style_disc":"false",
    "sample_z":"uniform",
    "add_D_noise":"false",
    "with_translation":"false",
    "with_scale":"false",
    "output_dir": "./HoloGAN"
     """

def load_dataset(args):
    train_loader = 0
    test_loader = 0
    return train_loader, test_loader

# Following preprocessing is taken from the original paper and modified a little for our's
def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True):
  image = load_webp(image_path)
  print(image.shape)
  return transform(image, input_height, input_width,
                   resize_height, resize_width, crop)

def load_webp(img_path):
    im = Image.open(img_path)
    return np.asarray(im)

def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    cropped_image = x[j:j+crop_h, i:i+crop_w]
    return np.array(Image.fromarray(cropped_image).resize((resize_w, resize_h)))

def transform(image, input_height, input_width,
              resize_height=64, resize_width=64, crop=True):
    if crop:
        cropped_image = center_crop(image, input_height, input_width, resize_height, resize_width)
    else:
        cropped_image = np.array(Image.fromarray(image).resize((resize_width, resize_height)))
    if len(cropped_image.shape) != 3: #In case of binary mask with no channels:
        cropped_image = np.expand_dims(cropped_image, -1)
    return np.array(cropped_image)[:, :, :3]/127.5 - 1.
