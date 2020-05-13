import numpy as np
import torch
from utils import initializer
from utils import load_dataset
import matplotlib.pyplot as plt

def main():
    model, optimizer, args = initializer()

    train_loader = load_dataset(args)
    for epoch in range(1, args.max_epochs + 1):
        print('{:3d}: '.format(epoch), end='')
        train(args, model, train_loader, optimizer)



def train(args, model, train_loader, optimizer):
    """train the model.

    Arguments are
    * args:         command line arguments entered by user.
    * model:        convolutional neural network model.
    * train_loader: train data loader.
    * optimizer:    optimize the model during training.

    This trains the model and prints the results of each epochs.
    """
    losses = []
    model.train()
    for data, _ in train_loader:
        plt.imshow(np.array(data[0].permute(1, 2, 0) * 255).astype(np.uint8))
        #data = data.to(args.device)
        #optimizer.zero_grad()

        #pred = model(data)
        #print(pred)
        #loss = F.cross_entropy(pred, target)

        #losses.append(float(loss))
        #loss.backward()
        #optimizer.step()
    plt.show()

    return


if __name__ == '__main__':
    main()
