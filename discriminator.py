"""
HoloGAN Discriminator implementation in PyTorch
May 17, 2020
"""
from torch import nn

class BasicBlock(nn.Module):
    """Basic Block defition of the Discriminator.
    """
    def __init__(self, in_planes, out_planes):
        super(BasicBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=2, padding=2)
        nn.init.normal_(self.conv2d.weight, std=0.02)
        self.conv2d_spec_norm = nn.utils.spectral_norm(self.conv2d)
        self.instance_norm = nn.InstanceNorm2d(out_planes)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.conv2d_spec_norm(x)
        out = self.instance_norm(out)
        out = self.lrelu(out)
        return out

class Discriminator(nn.Module):
    """Discriminator of HoloGAN
    """
    def __init__(self, in_planes, out_planes, z_planes):
        super(Discriminator, self).__init__()
        self.conv2d = nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=2, padding=2)
        nn.init.normal_(self.conv2d.weight, std=0.02)

        self.lrelu = nn.LeakyReLU(0.2)
        self.blocks = nn.Sequential(
            BasicBlock(out_planes, out_planes*2),
            BasicBlock(out_planes*2, out_planes*4),
            BasicBlock(out_planes*4, out_planes*8)
        )

        self.linear1 = nn.Linear(8192, 1)
        nn.init.normal_(self.linear1.weight, std=0.02)
        nn.init.constant_(self.linear1.bias, val=0.0)

        self.linear2 = nn.Linear(8192, 128)
        nn.init.normal_(self.linear2.weight, std=0.02)
        nn.init.constant_(self.linear2.bias, val=0.0)

        self.linear3 = nn.Linear(128, z_planes)
        nn.init.normal_(self.linear3.weight, std=0.02)
        nn.init.constant_(self.linear3.bias, val=0.0)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        batch_size = x.size(0)
        x0 = self.lrelu(self.conv2d(x))
        x3 = self.blocks(x0)
        # Flatten
        x3 = x3.view(batch_size, -1)
        # Returning logits to determine whether the images are real or fake
        x4 = self.linear1(x3)
        # Recognition network for latent variables has an additional layer
        encoder = self.lrelu(self.linear2(x3))
        cont_vars = self.tanh(self.linear3(encoder))

        return self.sigmoid(x4), x4, cont_vars
