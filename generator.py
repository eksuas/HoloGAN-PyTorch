from torch import nn
import torch

# TODO: test edilmedi
class Z_Mapping(nn.Module):
    def __init__(self, z_dimension, output_channel):
        super(Z_Mapping, self).__init__()
        self.output_channel = output_channel
        self.linear1 = nn.Linear(z_dimension, output_channel * 2)
        nn.init.normal_(self.linear1.weight, std=0.02)
        nn.init.constant_(self.linear1.bias, val=0.0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.linear1(x))
        return out[:, :self.output_channel], out[:, self.output_channel:]


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


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

class BasicBlock(nn.Module):
    """Basic Block defition of the Generator.
    """
    def __init__(self, z_planes, gf_dim):
        super(BasicBlock, self).__init__()
        self.convTranspose3d = nn.ConvTranspose3d(512, gf_dim*2, kernel_size=3,
                                                  stride=2, output_padding=1, padding=1, bias=True)

        nn.init.normal_(self.convTranspose3d.weight, std=0.02)
        nn.init.constant_(self.convTranspose3d.bias, val=0.0)

        self.z_mapping1 = Z_Mapping(z_planes, gf_dim*1)

    def forward(self, h, z):
        h = self.convTranspose3d(h)
        s, b = self.z_mapping2(z)
        h = AdaIn(h, s, b)
        h = self.relu(h)
        return h

class Generator(nn.Module):
    def __init__(self, z_planes, gf_dim, c_dim, reuse=False):
        super(Generator, self).__init__()
        # TODO: aşağıdaki none lar batch i temsil ediyor, forward da işimizi görür
        # self.view_in = tf.placeholder(tf.float32, [None, 6], name='view_in')
        # self.z = tf.placeholder(tf.float32, [None, cfg['z_dim']], name='z')

        # z = 128 or args.z_dim
        view_in = 6

        # batch_size = tf.shape(z)[0]
        # TODO: bu kısıma gerçekten gerek var mı emin değilim bunlar forward ile halledilebilir gibi
        s_h, s_w, s_d = 64, 64, 64

        # TODO: burada float değerler olabilirdi, ama biz sadece goal doğrultusunda bir convensionı destekliyoruz
        # TODO: .to(args.device) eklenmeli çok önemli cuda training için

        self.w16 = torch.empty((gf_dim * 8, s_h//16, s_w//16, s_d//16)).normal_(std=0.02)
        self.z_mapping = Z_Mapping(z_planes, gf_dim*8)
        self.relu = nn.ReLU(inplace=True)

        self.convTranspose3d2 = nn.ConvTranspose3d(512, gf_dim*2, kernel_size=3,
                                                  stride=2, output_padding=1, padding=1, bias=True)
        nn.init.normal_(self.convTranspose3d2.weight, std=0.02)
        nn.init.constant_(self.convTranspose3d2.bias, val=0.0)
        self.z_mapping2 = Z_Mapping(z_planes, gf_dim*2)


        self.convTranspose3d1 = nn.ConvTranspose3d(128, gf_dim*1, kernel_size=3,
                                                  stride=2, output_padding=1, padding=1, bias=True)
        nn.init.normal_(self.convTranspose3d1.weight, std=0.02)
        nn.init.constant_(self.convTranspose3d1.bias, val=0.0)
        self.z_mapping1 = Z_Mapping(z_planes, gf_dim*1)

        self.convTranspose2d16 = nn.ConvTranspose2d(1024, gf_dim*16, kernel_size=1,
                                                  stride=1, bias=True)
        nn.init.normal_(self.convTranspose2d16.weight, std=0.02)
        nn.init.constant_(self.convTranspose2d16.bias, val=0.0)


        self.convTranspose2d4 = nn.ConvTranspose2d(1024, gf_dim*4, kernel_size=4,
                                                  stride=2, padding=1, bias=True)
        nn.init.normal_(self.convTranspose2d4.weight, std=0.02)
        nn.init.constant_(self.convTranspose2d4.bias, val=0.0)


        self.convTranspose2d1 = nn.ConvTranspose2d(256, gf_dim*1, kernel_size=4,
                                                  stride=2, padding=1, bias=True)
        nn.init.normal_(self.convTranspose2d1.weight, std=0.02)
        nn.init.constant_(self.convTranspose2d1.bias, val=0.0)

        self.convTranspose2d = nn.ConvTranspose2d(64, c_dim, kernel_size=4,
                                                  stride=1, padding=1, bias=True)
        nn.init.normal_(self.convTranspose2d.weight, std=0.02)
        nn.init.constant_(self.convTranspose2d.bias, val=0.0)


        self.z_mapping1 = Z_Mapping(z_planes, gf_dim*1)
        self.z_mapping4 = Z_Mapping(z_planes, gf_dim*4)
        self.z_mapping8 = Z_Mapping(z_planes, gf_dim*8)

        self.tanh = nn.Tanh()

    def forward(self, x):
        # bu kısım backward phase da çalışabilir mi emin değilim
        batch_size = x.shape[0]
        w_tile = self.w16.reshape(tuple([1] + list(self.w16.shape)))
        new_shape = list(w_tile.shape)
        new_shape[0] *= batch_size
        new_shape = tuple(new_shape)
        w_tile = w_tile.repeat(batch_size, 1, 1, 1, 1).view(new_shape)

        s0, b0 = self.z_mapping8(x)
        h0 = AdaIn(w_tile, s0, b0)
        h0 = self.relu(h0)

        h1 = self.convTranspose3d2(h0)
        s1, b1 = self.z_mapping2(x)
        h1 = AdaIn(h1, s1, b1)
        h1 = self.relu(h1)

        h2 = self.convTranspose3d1(h1)
        s2, b2 = self.z_mapping1(x)
        h2 = AdaIn(h2, s2, b2)
        h2 = self.relu(h2)

        # h2_rotated = tf_3D_transform(h2, view_in, 16, 16)
        # h2_rotated = transform_voxel_to_match_image(h2_rotated)

        h2_rotated = h2
        h2_2d = h2.reshape(batch_size, -1, 16, 16)
        h3 = self.convTranspose2d16(h2_2d)
        h3 = self.relu(h3)

        h4 = self.convTranspose2d4(h3)
        s4, b4 = self.z_mapping4(x)
        h4 = AdaIn(h4, s4, b4)
        h4 = self.relu(h4)

        h5 = self.convTranspose2d1(h4)
        s5, b5 = self.z_mapping1(x)
        h5 = AdaIn(h5, s5, b5)
        h5 = self.relu(h5)

        print("h5.shape", h5.shape)
        h6 = self.convTranspose2d(h5)
        h6 = self.tanh(h6)

        print("h6.shape", h6.shape)
        return h6
