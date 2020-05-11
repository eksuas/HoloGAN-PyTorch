import numpy as np
import torch
from utils import initializer
from utils import get_image
from hologan import HoloGAN

def main():
     args = initializer()
     model = HoloGAN(z_dim = args.z_dim)
     #train_loader, test_loader = load_dataset(args)
     image_path = "/home/edanur/Documents/CENG796/project/dataset/img_align_celeba/000001.jpg"
     get_image(image_path=image_path, input_height=218, input_width=178,
                   resize_height=64, resize_width=64,
                   crop=True)

if __name__ == '__main__':
    main()
