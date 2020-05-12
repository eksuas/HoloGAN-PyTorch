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
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.linear1(x))
        return out[:, :output_channel], out[:, output_channel:]


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
    last = features.shape[-1]
    interval = list(range(len(features.shape)))[1:-1]
    new_shape = tuple([first] + [1] * len(interval) + [last])
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

class Generator(nn.Module):
    def __init__(self, z, gf_dim, reuse=False):
        super(Generator, self).__init__()
        # TODO: aşağıdaki none lar batch i temsil ediyor, forward da işimizi görür
        # self.view_in = tf.placeholder(tf.float32, [None, 6], name='view_in')
        # self.z = tf.placeholder(tf.float32, [None, cfg['z_dim']], name='z')

        # z = 128 or args.z_dim
        view_in = 6

        # batch_size = tf.shape(z)[0]
        # TODO: bu kısıma gerçekten gerek var mı emin değilim bunlar forward ile halledilebilir gibi
        s_h, s_w, s_d = 64, 64, 64
        #s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
        #s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
        #s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
        #s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

        # TODO: burada float değerler olabilirdi, ama biz sadece goal doğrultusunda bir convensionı destekliyoruz
        # TODO: .to(args.device) eklenmeli çok önemli cuda training için

        self.w16 = torch.empty((s_h//16, s_w//16, s_d//16, gf_dim * 8)).normal_(std=0.02)

        self.z_mapping = Z_Mapping(z, gf_dim*8)

    def forward(self, x):
        # bundan emin değilim test edelim
        batch_size = x.shape[0]
        new_shape = tuple([1] + list(self.w16.shape))
        w_tile = self.w16.reshape(new_shape)
        new_shape = list(w_tile.shape)
        new_shape[0] *= batch_size
        new_shape = tuple(new_shape)
        w_tile = w_tile.view(-1, 1).repeat(batch_size, 1, 1, 1, 1).view(new_shape)

        s0, b0 = self.z_mapping(x)
        h0 = AdaIn(w_tile, s0, b0)
        h0 = nn.Relu(h0)

        return h0

    """
      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('g_w_constant'):
              w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
              w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1))
              s0, b0 = self.z_mapping_function(z, self.gf_dim * 8, 'g_z0')
              h0 = AdaIn(w_tile, s0, b0)
              h0 = tf.nn.relu(h0)

          h1= deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2], k_h=3, k_d=3, k_w=3, name='g_h1')
          s1, b1 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z1')
          h1 = AdaIn(h1, s1, b1)
          h1 = tf.nn.relu(h1)

          h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_d=3, k_w=3, name='g_h2')
          s2, b2 = self.z_mapping_function(z, self.gf_dim * 1, 'g_z2')
          h2 = AdaIn(h2, s2, b2)
          h2 = tf.nn.relu(h2)

          #=============================================================================================================
          h2_rotated = tf_3D_transform(h2, view_in, 16, 16)
          h2_rotated = transform_voxel_to_match_image(h2_rotated)
          #=============================================================================================================
          # Collapsing depth dimension
          h2_2d = tf.reshape(h2_rotated, [batch_size, s_h4, s_w4, 16 * self.gf_dim])
          # 1X1 convolution
          h3 = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3')
          h3 = tf.nn.relu(h3)
          #=============================================================================================================

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4], k_h=4, k_w=4, name='g_h4')
          s4, b4 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z4')
          h4  = AdaIn(h4, s4, b4)
          h4  = tf.nn.relu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4, name='g_h5')
          s5, b5 = self.z_mapping_function(z, self.gf_dim, 'g_z5')
          h5 = AdaIn(h5, s5, b5)
          h5 = tf.nn.relu(h5)

          h6 = deconv2d(h5, [batch_size, s_h, s_w, self.c_dim], k_h=4, k_w=4, d_h=1, d_w=1, name='g_h6')

          output = tf.nn.tanh(h6, name="output")
          return output
    """
