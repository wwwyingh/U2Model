import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
# gid=0
class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def loss(self,y_pred):
        if self.penalty == 'l1':
            dif = [torch.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad.mean()
# 示例使用
# loss_fn = Grad().loss
# # deformation = torch.randn(1, 2, 256, 256)  # 假设形变场的大小为 (batch_size, 2, height, width)
# img = torch.zeros(1, 2, 256, 256)  # 假设图像的大小为 (batch_size, 1, height, width)
# #
# loss =loss_fn(img)
# print("Regularization Loss:", loss.item())
import torch
import torch.nn.functional as F
import math
import numpy as np

class NCC:
    """
    Local (over window) normalized cross correlation loss for multi-channel inputs.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):
        # Ensure the inputs are floating point
        y_true = y_true.float()
        y_pred = y_pred.float()

        # Assume inputs are [batch_size, channels, H, W]
        batch_size, channels, H, W = y_true.shape
        ndims = len(y_true.shape) - 2
        assert ndims in [1, 2], "Volumes should be 1 or 2 dimensions. Found: %d" % ndims

        # Set window size
        win = [9] * ndims if self.win is None else self.win

        # Create the convolution filter
        # Using depthwise convolution (groups=channels) to apply the filter to each channel independently
        sum_filt = torch.ones([channels, 1, *win], device=y_pred.device)

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1,)
            padding = (pad_no,)
            conv_fn = F.conv1d
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
            conv_fn = F.conv2d
        else:
            raise NotImplementedError("Only 1D and 2D NCC are implemented.")

        # Compute squares and products
        I2 = y_true * y_true
        J2 = y_pred * y_pred
        IJ = y_true * y_pred

        # Sum within the window
        I_sum = conv_fn(y_true, sum_filt, stride=stride, padding=padding, groups=channels)
        J_sum = conv_fn(y_pred, sum_filt, stride=stride, padding=padding, groups=channels)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding, groups=channels)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding, groups=channels)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding, groups=channels)

        # Compute means
        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        # Compute cross-correlation
        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        # Compute NCC
        cc = cross * cross / (I_var * J_var + 1e-5)

        # Average over all channels and batch
        return -torch.mean(cc)


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class MutualInformation2D(torch.nn.Module):
    """
    Mutual Information for 2D Images
    Adapted for 2D images from VoxelMorph
    """

    def __init__(self, sigma_ratio=1, minval=0., maxval=1., num_bin=32,device="cuda:0"):
        super(MutualInformation2D, self).__init__()

        """Create bin centers"""
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = Variable(torch.linspace(minval, maxval, num_bin), requires_grad=False).to(device)
        num_bins = len(bin_centers)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio

        self.preterm = 1 / (2 * sigma ** 2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers

    def mi(self, y_true, y_pred):
        # Clamp values to range
        y_true=y_true*0.5+0.5
        y_pred=y_pred*0.5+0.5
        y_pred = torch.clamp(y_pred, 0., self.max_clip)
        y_true = torch.clamp(y_true, 0., self.max_clip)

        # Reshape images into 2D format and add a dimension for bin centers
        y_true = y_true.reshape(y_true.shape[0], -1)
        y_true = torch.unsqueeze(y_true, 2)
        y_pred = y_pred.reshape(y_pred.shape[0], -1)
        y_pred = torch.unsqueeze(y_pred, 2)

        nb_voxels = y_pred.shape[1]  # total num of voxels (pixels in this case)

        """Reshape bin centers"""
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o).to(y_pred.device)

        """compute image terms by approx. Gaussian dist."""
        I_a = torch.exp(- self.preterm * torch.square(y_true - vbc))
        I_a = I_a / torch.sum(I_a, dim=-1, keepdim=True)

        I_b = torch.exp(- self.preterm * torch.square(y_pred - vbc))
        I_b = I_b / torch.sum(I_b, dim=-1, keepdim=True)

        # Compute joint probability pab and marginal probabilities pa, pb
        pab = torch.bmm(I_a.permute(0, 2, 1), I_b)
        pab = pab / nb_voxels
        pa = torch.mean(I_a, dim=1, keepdim=True)
        pb = torch.mean(I_b, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)
        return -mi.mean()  # average across batch

    def forward(self, y_true, y_pred):
        return self.mi(y_true, y_pred)

if __name__ == '__main__':
    x= torch.randn(1,2, 256, 256)  # 假设图像的大小为 (batch_size, 1, height, width)
    y = torch.randn(1, 2, 256, 256)  # 假设图像的大小为 (batch_size, 1, height, width)
    
    pred=NCC().loss(x,x)
    print(pred)