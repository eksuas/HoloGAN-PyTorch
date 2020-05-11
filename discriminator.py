from torch import nn
from spect_norm import spectral_norm

class BasicBlock(nn.Module):
    """Basic Block defition of the Discriminator.
    """
    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        # TODO: şimdilik harici weight initializingi desteklemiyoruz
        self.conv2d = nn.Conv2d(inplanes, planes, kernel_size=4)
        nn.init.normal_(self.conv2d.weight, std=0.02)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

        # TODO: aşağıdaki iki method CycleGAN'dan alınmış
        # TODO: spectral_norm da bu kullanılmış initializer=tf.truncated_normal_initializer()
        # self.spectral_norm = spectral_norm()
        #self.instance_norm = nn.InstanceNorm2d()

    def forward(self, x):
        out = self.Conv2d(x)
        out = spectral_norm(out)
        out = nn.InstanceNorm2d(out)
        out = self.lrelu(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, inplanes, planes, cont_dim, reuse=False):
        super(Discriminator, self).__init__()
        # TODO: add d noise implement edilsin mi default false
        # TODO: tensorflow da variable scope diye birşey varmış ve bununla reuse variable yapılabiliyormuş bunu ekleyelim mi
        # weight_initializer_type=tf.random_normal_initializer(stddev=0.02)
        # TODO: torch.nn.init.normal_(layers.weight, std=0.02)

        self.conv2d = nn.Conv2d(inplanes, planes, kernel_size=4)
        nn.init.normal_(self.conv2d.weight, std=0.02)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.block1 = BasicBlock(planes,   planes*2)
        self.block2 = BasicBlock(planes*2, planes*4)
        self.block3 = BasicBlock(planes*4, planes*8)

        self.linear1 = nn.Linear(planes*8, 1)
        nn.init.normal_(self.linear1.weight, std=0.02)
        nn.init.constant_(self.linear1.bias, val=0.0)

        self.linear2 = nn.Linear(planes*8, 128)
        nn.init.normal_(self.linear2.weight, std=0.02)
        nn.init.constant_(self.linear2.bias, val=0.0)

        self.linear3 = nn.Linear(128, cont_dim)
        nn.init.normal_(self.linear3.weight, std=0.02)
        nn.init.constant_(self.linear3.bias, val=0.0)

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
        cont_vars = self.linear3(encoder)

        return nn.Sigmoid(x4), x4, nn.Tanh(cont_vars)
