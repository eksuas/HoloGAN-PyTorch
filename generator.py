from torch import nn
import torch

class ZMapping(nn.Module):
    def __init__(self, z_dimension, output_channel):
        super(ZMapping, self).__init__()
        self.output_channel = output_channel
        self.linear1 = nn.Linear(z_dimension, output_channel * 2)
        nn.init.normal_(self.linear1.weight, std=0.02)
        nn.init.constant_(self.linear1.bias, val=0.0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.linear1(x))
        return out[:, :self.output_channel], out[:, self.output_channel:]

class BasicBlock(nn.Module):
    """Basic Block defition of the Generator.
    """
    def __init__(self, z_planes, in_planes, out_planes, transpose_dim):
        super(BasicBlock, self).__init__()
        if transpose_dim == 2:
            self.convTranspose = nn.ConvTranspose2d(in_planes,out_planes, kernel_size=4,
                                                    stride=2, padding=1, bias=True)
        else:
            self.convTranspose = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=3,
                                                    stride=2, output_padding=1, padding=1,
                                                    bias=True)

        nn.init.normal_(self.convTranspose.weight, std=0.02)
        nn.init.constant_(self.convTranspose.bias, val=0.0)
        self.zMapping = ZMapping(z_planes, out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, h, z):
        h = self.convTranspose(h)
        s, b = self.zMapping(z)
        h = AdaIn(h, s, b)
        h = self.relu(h)
        return h

# algoritması test edilerek geliştirildi
# TODO: ama birebir değerler ile test edilmedi
def AdaIn(features, scale, bias):
    """
    Adaptive instance normalization component. Works with both 4D and 5D tensors
    :features: features to be normalized
    :scale: scaling factor. This would otherwise be calculated as the sigma from a "style" features in style transfer
    :bias: bias factor. This would otherwise be calculated as the mean from a "style" features in style transfer
    """
    # TODO: bu kadar zahmete gerek var mıymış emin değilim
    # TODO: ileri de bu kısım için normalization ve linear layer eklenebilir!
    # if feature is 4D, the interval will be [1, 2]
    # if feature is 5D, the interval will be [1, 2, 3]
    first = features.shape[0]
    last = features.shape[1]
    interval = list(range(len(features.shape)))[2:]

    new_shape = tuple(list(features.shape)[:2] + [1] * len(interval))
    mean = features.mean(interval).reshape(new_shape)
    variance = features.var(interval).reshape(new_shape)

    sigma = torch.rsqrt(variance + 1e-8)
    normalized = (features - mean) * sigma
    scale_broadcast = scale.reshape(mean.shape)
    bias_broadcast = bias.reshape(mean.shape)
    normalized = scale_broadcast * normalized
    normalized += bias_broadcast
    return normalized

class Generator(nn.Module):
    def __init__(self, in_planes, out_planes, z_planes):
        super(Generator, self).__init__()
        # TODO: aşağıdaki none lar batch i temsil ediyor, forward da işimizi görür
        # self.view_in = tf.placeholder(tf.float32, [None, 6], name='view_in')
        view_in = 6

        # TODO: burada float değerler olabilirdi, ama biz sadece goal doğrultusunda bir convensionı destekliyoruz
        # TODO: .to(args.device) eklenmeli çok önemli cuda training için
        self.weight = torch.empty((in_planes*8, 4, 4, 4)).normal_(std=0.02)
        self.zMapping = ZMapping(z_planes, in_planes*8)
        self.block1 = BasicBlock(z_planes, in_planes=512,  out_planes=in_planes*2, transpose_dim=3)
        self.block2 = BasicBlock(z_planes, in_planes=128,  out_planes=in_planes, transpose_dim=3)
        self.block3 = BasicBlock(z_planes, in_planes=1024, out_planes=in_planes*4, transpose_dim=2)
        self.block4 = BasicBlock(z_planes, in_planes=256,  out_planes=in_planes, transpose_dim=2)

        self.convTranspose2d1 = nn.ConvTranspose2d(1024, in_planes*16, kernel_size=1,
                                                   stride=1, bias=True)
        nn.init.normal_(self.convTranspose2d1.weight, std=0.02)
        nn.init.constant_(self.convTranspose2d1.bias, val=0.0)

        self.convTranspose2d2 = nn.ConvTranspose2d(64, out_planes, kernel_size=4,
                                                   stride=1, padding=1, bias=True)
        nn.init.normal_(self.convTranspose2d2.weight, std=0.02)
        nn.init.constant_(self.convTranspose2d2.bias, val=0.0)

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # bu kısım backward phase da çalışabilir mi emin değilim
        batch_size = x.shape[0]
        w_tile = self.weight.unsqueeze(0).repeat(batch_size,1,1,1,1)
        s0, b0 = self.zMapping(x)
        h0 = AdaIn(w_tile, s0, b0)
        h0 = self.relu(h0)
        h1 = self.block1(h0, x)
        h2 = self.block2(h1, x)

        # h2_rotated = tf_3D_transform(h2, view_in, 16, 16)
        # h2_rotated = transform_voxel_to_match_image(h2_rotated)

        h2_rotated = h2
        h2_2d = h2.reshape(batch_size, -1, 16, 16)
        h3 = self.convTranspose2d1(h2_2d)
        h3 = self.relu(h3)
        h4 = self.block3(h3, x)
        h5 = self.block4(h4, x)
        h6 = self.convTranspose2d2(h5)
        h6 = self.tanh(h6)
        return h6
