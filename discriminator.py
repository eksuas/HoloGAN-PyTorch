from torch import nn
from utils import spectral_norm

class BasicBlock(nn.Module):
    """Basic Block defition of the Discriminator.
    """
    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        self.conv2d = nn.Conv2d(inplanes, planes, kernel_size=5, stride=2, padding=2)
        nn.init.normal_(self.conv2d.weight, std=0.02)

        self.conv2d_specNorm = nn.utils.spectral_norm(self.conv2d, self.conv2d.weight)
        self.instanceNorm = nn.InstanceNorm2d(planes)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.conv2d_specNorm(x)
        out = self.instanceNorm(out)
        out = self.lrelu(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, inplanes, planes, cont_dim, reuse=False):
        super(Discriminator, self).__init__()
        self.conv2d = nn.Conv2d(inplanes, planes, kernel_size=5, stride=2, padding=2)
        nn.init.normal_(self.conv2d.weight, std=0.02)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.block1 = BasicBlock(planes,   planes*2)
        self.block2 = BasicBlock(planes*2, planes*4)
        self.block3 = BasicBlock(planes*4, planes*8)

        self.linear1 = nn.Linear(8192, 1)
        nn.init.normal_(self.linear1.weight, std=0.02)
        nn.init.constant_(self.linear1.bias, val=0.0)

        self.linear2 = nn.Linear(8192, 128)
        nn.init.normal_(self.linear2.weight, std=0.02)
        nn.init.constant_(self.linear2.bias, val=0.0)

        self.linear3 = nn.Linear(128, cont_dim)
        nn.init.normal_(self.linear3.weight, std=0.02)
        nn.init.constant_(self.linear3.bias, val=0.0)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x0 = self.lrelu(self.conv2d(x))
        x1 = self.block1(x0)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        # Flatten
        x3 = x3.view(x3.size(0), -1)
        # Returning logits to determine whether the images are real or fake
        x4 = self.linear1(x3)
        # Recognition network for latent variables has an additional layer
        encoder = self.lrelu(self.linear2(x3))
        cont_vars = self.tanh(self.linear3(encoder))

        return self.sigmoid(x4), x4, cont_vars
