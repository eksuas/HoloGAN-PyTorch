from torch import nn
import torch

class ZMapping(nn.Module):
    def __init__(self, z_dimension, output_channel):
        super(ZMapping, self).__init__()
        self.output_channel = output_channel
        self.linear1 = nn.Linear(z_dimension, output_channel * 2)
        nn.init.normal_(self.linear1.weight, std=0.02)
        nn.init.constant_(self.linear1.bias, val=0.0)
        self.relu = nn.ReLU()

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
        self.relu = nn.ReLU()

    def forward(self, h, z):
        h = self.convTranspose(h)
        s, b = self.zMapping(z)
        h = AdaIn(h, s, b)
        h = self.relu(h)
        return h

# algoritması test edilerek geliştirildi
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


def meshgrid(height, width, depth):
    z, y, x = torch.meshgrid(torch.arange(depth), torch.arange(height), torch.arange(width))
    x_flat = x.reshape(1, -1).float()
    y_flat = y.reshape(1, -1).float()
    z_flat = z.reshape(1, -1).float()
    ones = torch.ones(x_flat.shape)
    return torch.cat([x_flat, y_flat, z_flat, ones], axis=0)


def interpolation(voxel, x, y, z, out_shape):

    batch_size = voxel.shape[0]
    n_channels = voxel.shape[1]
    height     = voxel.shape[2]
    width      = voxel.shape[3]
    depth      = voxel.shape[4]

    x = x.float()
    y = y.float()
    z = z.float()

    out_channel = out_shape[1]
    out_height  = out_shape[2]
    out_width   = out_shape[3]
    out_depth   = out_shape[4]

    max_y = height - 1
    max_x = width  - 1
    max_z = depth  - 1

    # do sampling
    x0 = torch.floor(x).int()
    x1 = x0 + 1
    y0 = torch.floor(y).int()
    y1 = y0 + 1
    z0 = torch.floor(z).int()
    z1 = z0 + 1

    x0 = torch.clamp(x0, 0, max_x)
    x1 = torch.clamp(x1, 0, max_x)
    y0 = torch.clamp(y0, 0, max_y)
    y1 = torch.clamp(y1, 0, max_y)
    z0 = torch.clamp(z0, 0, max_z)
    z1 = torch.clamp(z1, 0, max_z)

    x = torch.arange(batch_size) * width * height * depth
    rep = torch.ones(1, out_height * out_width * out_depth).long()
    base = torch.matmul(x.reshape(-1, 1), rep).reshape(-1)

    #Find the Z element of each index
    base_z0 = base + z0 * width * height
    base_z1 = base + z1 * width * height

    #Find the Y element based on Z
    base_z0_y0 = base_z0 + y0 * width
    base_z0_y1 = base_z0 + y1 * width
    base_z1_y0 = base_z1 + y0 * width
    base_z1_y1 = base_z1 + y1 * width

    # Find the X element based on Y, Z for Z=0
    idx_a = (base_z0_y0 + x0).long()
    idx_b = (base_z0_y1 + x0).long()
    idx_c = (base_z0_y0 + x1).long()
    idx_d = (base_z0_y1 + x1).long()

    # Find the X element based on Y,Z for Z =1
    idx_e = (base_z1_y0 + x0).long()
    idx_f = (base_z1_y1 + x0).long()
    idx_g = (base_z1_y0 + x1).long()
    idx_h = (base_z1_y1 + x1).long()

    # use indices to lookup pixels in the flat image and restore channels dim
    voxel_flat = voxel.reshape(-1, n_channels).float()
    Ia = voxel_flat[idx_a]
    Ib = voxel_flat[idx_b]
    Ic = voxel_flat[idx_c]
    Id = voxel_flat[idx_d]
    Ie = voxel_flat[idx_e]
    If = voxel_flat[idx_f]
    Ig = voxel_flat[idx_g]
    Ih = voxel_flat[idx_h]

    # and finally calculate interpolated values
    x0_f = x0.float()
    x1_f = x1.float()
    y0_f = y0.float()
    y1_f = y1.float()
    z0_f = z0.float()
    z1_f = z1.float()

    #First slice XY along Z where z=0
    wa = ((x1_f - x) * (y1_f - y) * (z1_f - z)).unsqueeze(1)
    wb = ((x1_f - x) * (y - y0_f) * (z1_f - z)).unsqueeze(1)
    wc = ((x - x0_f) * (y1_f - y) * (z1_f - z)).unsqueeze(1)
    wd = ((x - x0_f) * (y - y0_f) * (z1_f - z)).unsqueeze(1)

    # First slice XY along Z where z=1
    we = ((x1_f - x) * (y1_f - y) * (z - z0_f)).unsqueeze(1)
    wf = ((x1_f - x) * (y - y0_f) * (z - z0_f)).unsqueeze(1)
    wg = ((x - x0_f) * (y1_f - y) * (z - z0_f)).unsqueeze(1)
    wh = ((x - x0_f) * (y - y0_f) * (z - z0_f)).unsqueeze(1)

    target = sum([wa * Ia, wb * Ib, wc * Ic, wd * Id,  we * Ie, wf * If, wg * Ig, wh * Ih])
    return target.reshape(out_shape)


def apply_transformation(voxel_array, transformation_matrix, size=64, new_size=128):

    batch_size = voxel_array.shape[0]
    target = torch.zeros([batch_size, new_size, new_size, new_size])
    # Aligning the centroid of the object (voxel grid) to origin for rotation,
    # then move the centroid back to the original position of the grid centroid
    centroid = torch.tensor([[1,0,0, -size * 0.5],
                             [0,1,0, -size * 0.5],
                             [0,0,1, -size * 0.5],
                             [0,0,0,           1]])
    centroid = centroid.reshape(1, 4, 4).repeat(batch_size, 1, 1)

    # However, since the rotated grid might be out of bound for the original grid size,
    # move the rotated grid to a new bigger grid
    centroid_new = torch.tensor([[1, 0, 0, new_size * 0.5],
                                     [0, 1, 0, new_size * 0.5],
                                     [0, 0, 1, new_size * 0.5],
                                     [0, 0, 0,              1]])
    centroid_new = centroid_new.reshape(1, 4, 4).repeat(batch_size, 1, 1)

    transformed_centoid = torch.matmul(torch.matmul(centroid_new, transformation_matrix), centroid)

    # TODO: devam eden iki işlemi neden yapıyoruz anlayamadım
    transformed_centoid = transformed_centoid.inverse()
    transformed_centoid = transformed_centoid[:, 0:3, :] #Ignore the homogenous coordinate so the results are 3D vectors

    grid = meshgrid(new_size, new_size, new_size)
    grid = grid.reshape(1, grid.shape[0], grid.shape[1])
    grid = grid.repeat(batch_size, 1, 1)

    grid_transform = torch.matmul(transformed_centoid, grid)
    x_flat = grid_transform[:, 0, :].reshape(-1)
    y_flat = grid_transform[:, 1, :].reshape(-1)
    z_flat = grid_transform[:, 2, :].reshape(-1)

    n_channels = voxel_array.shape[1]
    out_shape = (batch_size, n_channels, new_size, new_size, new_size)
    transformed = interpolation(voxel_array, x_flat, y_flat, z_flat, out_shape).reshape(out_shape)
    return transformed


def transformation3d(voxel_array, view_params, size=64, new_size=128):
    # TODO: daha efficient olması için ileri de tek bir matrix formatında oluşturabilir

    # Rotation Y matrix
    theta = torch.tensor(view_params[:, 0].reshape(-1, 1, 1))
    gamma = torch.tensor(view_params[:, 1].reshape(-1, 1, 1))
    ones = torch.ones(theta.shape)
    zeros = torch.zeros(theta.shape)
    rot_y = torch.cat([
        torch.cat([theta.cos().float(),  zeros,  -theta.sin().float(),  zeros], axis=2),
        torch.cat([zeros,                ones,   zeros,                 zeros], axis=2),
        torch.cat([theta.sin().float(),  zeros,  theta.cos().float(),   zeros], axis=2),
        torch.cat([zeros,                zeros,  zeros,                 ones],  axis=2)], axis=1)

    # Rotation Z matrix
    rot_z = torch.cat([
        torch.cat([gamma.cos().float(),  gamma.sin().float(),   zeros,  zeros], axis=2),
        torch.cat([-gamma.sin().float(), gamma.cos().float(),   zeros,  zeros], axis=2),
        torch.cat([zeros,                zeros,                 ones,   zeros], axis=2),
        torch.cat([zeros,                zeros,                 zeros,  ones],  axis=2)], axis=1)

    rotation_matrix = torch.matmul(rot_z, rot_y)

    # Scaling matrix
    scale = torch.tensor(view_params[:, 2].reshape(-1, 1, 1)).float()
    scaling_matrix = torch.cat([
        torch.cat([scale, zeros,  zeros, zeros], axis=2),
        torch.cat([zeros, scale,  zeros, zeros], axis=2),
        torch.cat([zeros, zeros,  scale, zeros], axis=2),
        torch.cat([zeros, zeros,  zeros, ones],  axis=2)], axis=1)

    # Translation matrix
    x_shift = torch.tensor(view_params[:,3].reshape(-1, 1, 1)).float()
    y_shift = torch.tensor(view_params[:,4].reshape(-1, 1, 1)).float()
    z_shift = torch.tensor(view_params[:,5].reshape(-1, 1, 1)).float()
    translation_matrix = torch.cat([
        torch.cat([ones,  zeros, zeros, x_shift], axis=2),
        torch.cat([zeros, ones,  zeros, y_shift], axis=2),
        torch.cat([zeros, zeros, ones,  z_shift], axis=2),
        torch.cat([zeros, zeros, zeros, ones],    axis=2)], axis=1)

    transformation_matrix = torch.matmul(translation_matrix, scaling_matrix)
    transformation_matrix = torch.matmul(transformation_matrix, rotation_matrix)

    return apply_transformation(voxel_array, transformation_matrix, size, new_size)


class Generator(nn.Module):
    def __init__(self, in_planes, out_planes, z_planes, view_planes=6):
        super(Generator, self).__init__()
        self.weight = torch.empty((in_planes*8, 4, 4, 4)).normal_(std=0.02)

        self.zMapping = ZMapping(z_planes, in_planes*8)
        self.block1 = BasicBlock(z_planes, in_planes=in_planes*8,  out_planes=in_planes*2, transpose_dim=3)
        self.block2 = BasicBlock(z_planes, in_planes=in_planes*2,  out_planes=in_planes,   transpose_dim=3)

        self.convTranspose2d1 = nn.ConvTranspose2d(in_planes*16, in_planes*16, kernel_size=1)
        nn.init.normal_(self.convTranspose2d1.weight, std=0.02)
        nn.init.constant_(self.convTranspose2d1.bias, val=0.0)

        self.block3 = BasicBlock(z_planes, in_planes=in_planes*16, out_planes=in_planes*4, transpose_dim=2)
        self.block4 = BasicBlock(z_planes, in_planes=in_planes*4,  out_planes=in_planes,   transpose_dim=2)

        self.convTranspose2d2 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, padding=1)
        nn.init.normal_(self.convTranspose2d2.weight, std=0.02)
        nn.init.constant_(self.convTranspose2d2.bias, val=0.0)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, z, view_in):
        batch_size = z.shape[0]
        #print("batch_size:", batch_size)
        w_tile = self.weight.unsqueeze(0).repeat(batch_size,1,1,1,1)
        #print("w_tile.shape:", w_tile.shape)
        s0, b0 = self.zMapping(z)
        #print("s0.shape:", s0.shape)
        #print("b0.shape:", b0.shape)
        h0 = AdaIn(w_tile, s0, b0)
        h0 = self.relu(h0)
        #print("h0.shape:", h0.shape)
        h1 = self.block1(h0, z)
        #print("h1.shape:", h1.shape)
        h2 = self.block2(h1, z)
        #print("h2.shape:", h2.shape)

        h2_rotated = transformation3d(h2, view_in, 16, 16)
        #print("h2_rotated.shape:", h2_rotated.shape)
        h2_rotated = h2_rotated.permute(0, 1, 3, 2, 4)
        inv_idx = torch.arange(h2_rotated.size(2)-1, -1, -1).long()
        h2_rotated = h2_rotated[:, :, inv_idx, :, :]
        #print("h2_rotated.shape:", h2_rotated.shape)

        h2_2d = h2_rotated.reshape(batch_size, -1, 16, 16)
        #print("h2_2d.shape:", h2_2d.shape)
        h3 = self.convTranspose2d1(h2_2d)
        h3 = self.relu(h3)
        #print("h3.shape:", h3.shape)
        h4 = self.block3(h3, z)
        #print("h4.shape:", h4.shape)
        h5 = self.block4(h4, z)
        #print("h5.shape:", h5.shape)
        h6 = self.convTranspose2d2(h5)
        h6 = self.tanh(h6)
        #print("h6.shape:", h6.shape)
        return h6
