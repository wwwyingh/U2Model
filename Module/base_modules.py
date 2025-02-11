import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.distributions.normal import Normal

class ConvInsBlock(nn.Module):
    """Convolutional block with InstanceNorm and LeakyReLU."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()
        self.main = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        return self.activation(self.norm(self.main(x)))


class ResBlock(nn.Module):
    """Residual Block with InstanceNorm and LeakyReLU activation."""
    def __init__(self, channels, alpha=0.1):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(alpha),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


class SpatialTransformer(nn.Module):
    """2D Spatial Transformer for applying transformations to images."""
    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.mode = mode
        grid = self.create_sampling_grid(size)
        self.register_buffer('grid', grid)

    @staticmethod
    def create_sampling_grid(size):
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        return grid.unsqueeze(0).type(torch.FloatTensor)

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 1)[..., [1, 0]]
        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):
    """Integrates a vector field via scaling and squaring."""
    def __init__(self, inshape, nsteps=7):
        super().__init__()
        assert nsteps >= 0, f'nsteps should be >= 0, found: {nsteps}'
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class UpConvBlock(nn.Module):
    """UpConvolutional block with ConvTranspose2d, InstanceNorm, and LeakyReLU."""
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, alpha=0.1):
        super(UpConvBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)
        self.actout = nn.Sequential(nn.InstanceNorm2d(out_channels), nn.LeakyReLU(alpha))

    def forward(self, x):
        return self.actout(self.upconv(x))


class FiLMModulation(nn.Module):
    """FiLM modulation layer for conditional feature modulation."""
    def __init__(self, input_dim=2, output_dim=2):
        super(FiLMModulation, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x, c):
        gamma, beta = torch.chunk(self.fc(c), 2, dim=1)
        gamma = gamma.view(gamma.size(0), gamma.size(1), 1, 1).expand_as(x)
        beta = beta.view(beta.size(0), beta.size(1), 1, 1).expand_as(x)
        return gamma * x + beta

class CConv(nn.Module):
    def __init__(self, channel):
        super(CConv, self).__init__()

        c = channel

        self.conv = nn.Sequential(
            ConvInsBlock(c, c, 3, 1),
            ConvInsBlock(c, c, 3, 1)
        )

    def forward(self, float_fm, fixed_fm, d_fm):
        concat_fm = torch.cat([float_fm, fixed_fm, d_fm], dim=1)
        x = self.conv(concat_fm)
        return x


class FiLMModulation_drop(nn.Module):
    def __init__(self, input_dim=2, output_dim=2, dropout_prob=0.3):
        super(FiLMModulation_drop, self).__init__()
        # 这里假设 γ 和 β 是通过一个全连接层从条件 c 中计算的
        self.fc = nn.Linear(input_dim, output_dim * 2)
        # 添加 Dropout 层
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, c):
        # c 的维度是 (bs, 2), 通过全连接层计算 γ 和 β
        gamma_beta = self.fc(c)  # 输出是 (bs, output_dim * 2)

        # 将 γ 和 β 分开
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)  # γ 和 β 都是 (bs, output_dim)

        # 应用 Dropout 到 gamma 和 beta 上
        gamma = self.dropout(gamma)  # 在训练时，gamma 会被随机丢弃部分神经元
        beta = self.dropout(beta)  # 同理，beta 也会应用 Dropout

        # 调整 γ 和 β 的维度，使其适用于输入 x
        gamma = gamma.view(gamma.size(0), gamma.size(1), 1, 1).expand(gamma.size(0), gamma.size(1), x.size(2), x.size(3))
        beta = beta.view(beta.size(0), beta.size(1), 1, 1).expand(beta.size(0), beta.size(1), x.size(2), x.size(3))

        # 对输入 x 进行调制: FiLM(x) = γ(c) * x + β(c)
        x_modulated = gamma * x + beta
        return x_modulated
