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


def spectral_norm(w, iteration=1, u_weight=None):
    w_shape = list(w.shape)
    w = w.reshape(-1, w_shape[-1])
    if u_weight is None:
        u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)
    else:
        u = u_weight

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm
