import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple


class Grad:
    """
    N-D Gradient Loss for deformation field regularization.
    Penalizes sudden changes in the displacement field.
    """

    def __init__(self, penalty: str = 'l1', loss_mult: Optional[float] = None) -> None:
        """
        Args:
            penalty: Either 'l1' or 'l2' for different gradient loss types.
            loss_mult: Optional multiplier for the loss.
        """
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _compute_diffs(self, y_pred: torch.Tensor) -> list:
        """
        Compute differences along each spatial dimension.
        """
        ndims = len(y_pred.shape) - 2  # Excluding batch and channels
        diffs = [y_pred.narrow(dim=i + 2, start=1, length=y_pred.shape[i + 2] - 1) -
                 y_pred.narrow(dim=i + 2, start=0, length=y_pred.shape[i + 2] - 1)
                 for i in range(ndims)]
        return diffs

    def loss(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient regularization loss.

        Args:
            y_pred: Predicted deformation field.

        Returns:
            Gradient loss.
        """
        diffs = self._compute_diffs(y_pred)

        if self.penalty == 'l1':
            diffs = [torch.abs(d) for d in diffs]
        elif self.penalty == 'l2':
            diffs = [d ** 2 for d in diffs]
        else:
            raise ValueError(f"Penalty must be 'l1' or 'l2', got {self.penalty}")

        loss = sum(torch.mean(d) for d in diffs) / len(diffs)
        return loss * self.loss_mult if self.loss_mult else loss


class NCC:
    """
    Local Normalized Cross-Correlation (NCC) Loss for image similarity measurement.
    """

    def __init__(self, win: Optional[int] = None) -> None:
        """
        Args:
            win: Window size for local similarity computation.
        """
        self.win = win if win else 9  # Default window size of 9x9

    def loss(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute NCC loss.

        Args:
            y_true: Ground truth image.
            y_pred: Predicted image.

        Returns:
            NCC loss.
        """
        y_true, y_pred = y_true.float(), y_pred.float()
        batch_size, channels, H, W = y_true.shape
        ndims = 2  # 2D images

        sum_filt = torch.ones((channels, 1, self.win, self.win), device=y_pred.device)
        stride = (1, 1)
        padding = (self.win // 2, self.win // 2)

        # Compute sums within window
        I_sum = F.conv2d(y_true, sum_filt, stride=stride, padding=padding, groups=channels)
        J_sum = F.conv2d(y_pred, sum_filt, stride=stride, padding=padding, groups=channels)
        I2_sum = F.conv2d(y_true ** 2, sum_filt, stride=stride, padding=padding, groups=channels)
        J2_sum = F.conv2d(y_pred ** 2, sum_filt, stride=stride, padding=padding, groups=channels)
        IJ_sum = F.conv2d(y_true * y_pred, sum_filt, stride=stride, padding=padding, groups=channels)

        win_size = self.win ** 2
        u_I, u_J = I_sum / win_size, J_sum / win_size
        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I ** 2 * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J ** 2 * win_size

        ncc = cross ** 2 / (I_var * J_var + 1e-6)  # Add small constant for numerical stability
        return -torch.mean(ncc)


class MutualInformation2D(nn.Module):
    """
    Mutual Information Loss for 2D Images.
    """

    def __init__(self, sigma_ratio: float = 1, minval: float = 0., maxval: float = 1., num_bin: int = 32, device: str = "cuda:0") -> None:
        """
        Args:
            sigma_ratio: Ratio for Gaussian kernel.
            minval: Minimum value for intensity bins.
            maxval: Maximum value for intensity bins.
            num_bin: Number of bins for probability distribution.
            device: Device for computations.
        """
        super(MutualInformation2D, self).__init__()

        self.num_bins = num_bin
        self.device = device
        self.max_clip = maxval
        self.vol_bin_centers = torch.linspace(minval, maxval, num_bin, device=device)
        sigma = np.mean(np.diff(self.vol_bin_centers.cpu().numpy())) * sigma_ratio
        self.preterm = 1 / (2 * sigma ** 2)

    def _compute_probability(self, y: torch.Tensor) -> torch.Tensor:
        """
        Compute probability distribution over bins using a Gaussian model.

        Args:
            y: Input tensor.

        Returns:
            Probability tensor.
        """
        y = torch.clamp(y * 0.5 + 0.5, 0., self.max_clip)  # Normalize to [0,1]
        y = y.view(y.shape[0], -1, 1)
        bin_centers = self.vol_bin_centers.view(1, 1, -1)
        prob = torch.exp(-self.preterm * (y - bin_centers) ** 2)
        return prob / torch.sum(prob, dim=-1, keepdim=True)

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute mutual information loss.

        Args:
            y_true: Ground truth image.
            y_pred: Predicted image.

        Returns:
            Mutual information loss.
        """
        I_a = self._compute_probability(y_true)
        I_b = self._compute_probability(y_pred)
        pab = torch.bmm(I_a.permute(0, 2, 1), I_b) / y_true.shape[1]
        pa, pb = torch.mean(I_a, dim=1, keepdim=True), torch.mean(I_b, dim=1, keepdim=True)
        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6

        mi = torch.sum(pab * torch.log(pab / papb + 1e-6), dim=[1, 2])
        return -mi.mean()


if __name__ == '__main__':
    x = torch.randn(1, 1, 256, 256)  # Example input
    y = torch.randn(1, 1, 256, 256)

    grad_loss = Grad(penalty='l2').loss(x)
    ncc_loss = NCC().loss(x, y)
    mi_loss = MutualInformation2D().forward(x, y)

    print(f"Gradient Loss: {grad_loss.item():.6f}")
    print(f"NCC Loss: {ncc_loss.item():.6f}")
    print(f"Mutual Information Loss: {mi_loss.item():.6f}")
