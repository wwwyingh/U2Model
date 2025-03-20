import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.startswith('Conv'):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)

class INSResBlock(nn.Module):
    def conv3x3(self, inplanes, out_planes, stride=1):
        return [nn.ReflectionPad2d(1), nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride)]
    def __init__(self, inplanes, planes, stride=1, dropout=0.0):
        super(INSResBlock, self).__init__()
        model = []
        model += self.conv3x3(inplanes, planes, stride)
        model += [nn.InstanceNorm2d(planes)]
        model += [nn.ReLU(inplace=False)]
        model += self.conv3x3(planes, planes)
        model += [nn.InstanceNorm2d(planes)]
        if dropout > 0:
            model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)
    def forward(self, x):
        residual = x
        out = self.model(x)
        return out + residual

class ReLUINSConvTranspose2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding, output_padding):
        super(ReLUINSConvTranspose2d, self).__init__()
        model = []
        model += [nn.ConvTranspose2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding,
                                      output_padding=output_padding, bias=True)]
        model += [LayerNorm(n_out)]
        model += [nn.ReLU(inplace=False)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)
    def forward(self, x):
        return self.model(x)

class LayerNorm(nn.Module):
    def __init__(self, n_out, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.n_out = n_out
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(n_out, 1, 1))
            self.bias = nn.Parameter(torch.zeros(n_out, 1, 1))
    def forward(self, x):
        normalized_shape = x.size()[1:]
        if self.affine:
            return F.layer_norm(x, normalized_shape, self.weight.expand(normalized_shape), self.bias.expand(normalized_shape))
        else:
            return F.layer_norm(x, normalized_shape)

def conv3x3(in_planes, out_planes):
    return [nn.ReflectionPad2d(1), nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=0, bias=True)]

def meanpoolConv(inplanes, outplanes):
    sequence = [nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True)]
    return nn.Sequential(*sequence)

def convMeanpool(inplanes, outplanes):
    sequence = conv3x3(inplanes, outplanes) + [nn.AvgPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*sequence)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
        super(BasicBlock, self).__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += conv3x3(inplanes, inplanes)
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [convMeanpool(inplanes, outplanes)]
        self.conv = nn.Sequential(*layers)
        self.shortcut = meanpoolConv(inplanes, outplanes)
    def forward(self, x):
        return self.conv(x) + self.shortcut(x)

class LeakyReLUConv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0, norm='None', sn=False):
        super(LeakyReLUConv2d, self).__init__()
        model = []
        model += [nn.ReflectionPad2d(padding)]
        if sn:
            from torch.nn.utils import spectral_norm
            model += [spectral_norm(nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True))]
        else:
            model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]
        if norm == 'Instance':
            model += [nn.InstanceNorm2d(n_out, affine=False)]
        model += [nn.LeakyReLU(inplace=False)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)
    def forward(self, x):
        return self.model(x)

class ReLUINSConv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
        super(ReLUINSConv2d, self).__init__()
        model = [nn.ReflectionPad2d(padding),
                 nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True),
                 nn.InstanceNorm2d(n_out, affine=False),
                 nn.ReLU(inplace=False)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)
    def forward(self, x):
        return self.model(x)

class SpectralNorm:
    def __init__(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but got {}'.format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps
    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        weight_mat = weight
        if self.dim != 0:
            weight_mat = weight_mat.permute(self.dim, *[d for d in range(weight_mat.dim()) if d != self.dim])
        height = weight_mat.size(0)
        weight_mat = weight_mat.reshape(height, -1)
        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                v = F.normalize(torch.matmul(weight_mat.t(), u), dim=0, eps=self.eps)
                u = F.normalize(torch.matmul(weight_mat, v), dim=0, eps=self.eps)
        sigma = torch.dot(u, torch.matmul(weight_mat, v))
        weight = weight / sigma
        return weight, u
    def remove(self, module):
        weight = getattr(module, self.name)
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        delattr(module, self.name + '_orig')
        module.register_parameter(self.name, torch.nn.Parameter(weight))
    def __call__(self, module, inputs):
        if module.training:
            weight, u = self.compute_weight(module)
            setattr(module, self.name, weight)
            setattr(module, self.name + '_u', u)
        else:
            r_g = getattr(module, self.name + '_orig').requires_grad
            getattr(module, self.name).detach_().requires_grad_(r_g)
    @staticmethod
    def apply(module, name, n_power_iterations, dim, eps):
        fn = SpectralNorm(name, n_power_iterations, dim, eps)
        weight = module._parameters[name]
        height = weight.size(dim)
        u = F.normalize(weight.new_empty(height).normal_(0, 1), dim=0, eps=fn.eps)
        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        module.register_buffer(fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_forward_pre_hook(fn)
        return fn

def spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
    if dim is None:
        if isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    return module

class GaussianNoiseLayer(nn.Module):
    def __init__(self):
        super(GaussianNoiseLayer, self).__init__()
    def forward(self, x):
        if not self.training:
            return x
        noise = torch.randn_like(x)
        return x + noise

# Additional basic modules (from base_modules.py)

class ConvInsBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, alpha=0.1):
        super(ConvInsBlock, self).__init__()
        self.main = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.activation = nn.LeakyReLU(alpha)
    def forward(self, x):
        return self.activation(self.norm(self.main(x)))

class ResBlock(nn.Module):
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
    def __init__(self, size, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        self.mode = mode
        grid = self.create_sampling_grid(size)
        self.register_buffer('grid', grid)
    @staticmethod
    def create_sampling_grid(size):
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(*vectors, indexing='ij')
        grid = torch.stack(grids)
        return grid.unsqueeze(0).float()
    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 1)[..., [1, 0]]
        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class VecInt(nn.Module):
    def __init__(self, inshape, nsteps=7):
        super(VecInt, self).__init__()
        assert nsteps >= 0, "nsteps should be >= 0"
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** nsteps)
        self.transformer = SpatialTransformer(inshape)
    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, alpha=0.1):
        super(UpConvBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)
        self.actout = nn.Sequential(nn.InstanceNorm2d(out_channels), nn.LeakyReLU(alpha))
    def forward(self, x):
        return self.actout(self.upconv(x))

class FiLMModulation(nn.Module):
    def __init__(self, input_dim=2, output_dim=2):
        super(FiLMModulation, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)
    def forward(self, x, c):
        gamma, beta = torch.chunk(self.fc(c), 2, dim=1)
        gamma = gamma.view(gamma.size(0), gamma.size(1), 1, 1).expand_as(x)
        beta = beta.view(beta.size(0), beta.size(1), 1, 1).expand_as(x)
        return gamma * x + beta

class FiLMModulation_drop(nn.Module):
    def __init__(self, input_dim=2, output_dim=2, dropout_prob=0.3):
        super(FiLMModulation_drop, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)
        self.dropout = nn.Dropout(dropout_prob)
    def forward(self, x, c):
        gamma_beta = self.fc(c)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gamma = self.dropout(gamma)
        beta = self.dropout(beta)
        gamma = gamma.view(gamma.size(0), gamma.size(1), 1, 1).expand_as(x)
        beta = beta.view(beta.size(0), beta.size(1), 1, 1).expand_as(x)
        return gamma * x + beta

class CConv(nn.Module):
    def __init__(self, channel):
        super(CConv, self).__init__()
        c = channel
        self.conv = nn.Sequential(
            ConvInsBlock(c, c, kernel_size=3, stride=1),
            ConvInsBlock(c, c, kernel_size=3, stride=1)
        )
    def forward(self, float_fm, fixed_fm, d_fm):
        concat_fm = torch.cat([float_fm, fixed_fm, d_fm], dim=1)
        x = self.conv(concat_fm)
        return x
