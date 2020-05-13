import torch
import numpy as np
from torchvision import datasets, transforms

def load_dataset(args):
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
