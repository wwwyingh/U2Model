from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import comb
from typing import Tuple


class StochasticNonLinearIntensityTransformation(nn.Module):
    def __init__(self, delta: float = 0.5) -> None:
        super(StochasticNonLinearIntensityTransformation, self).__init__()
        self.delta = delta

    def _bezier_curve(self, control_points: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        n = control_points.size(0) - 1
        i = torch.arange(n + 1, device=t.device).view(-1, 1)
        binom_coeff = torch.tensor([comb(n, k) for k in range(n + 1)],
                                   device=t.device, dtype=t.dtype).view(-1, 1)
        bernstein_poly = binom_coeff * (t ** i) * ((1 - t) ** (n - i))
        curve = torch.mm(control_points[:, 1].view(1, -1), bernstein_poly).squeeze()
        return curve

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        min_val = image.view(image.size(0), -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        max_val = image.view(image.size(0), -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        return 2 * (image - min_val) / (max_val - min_val) - 1

    def _linear_interpolate(self, x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
        idx = torch.searchsorted(xp, x)
        idx = torch.clamp(idx, 1, len(xp) - 1)
        x0 = xp[idx - 1]
        x1 = xp[idx]
        y0 = fp[idx - 1]
        y1 = fp[idx]
        slope = (y1 - y0) / (x1 - x0)
        return y0 + slope * (x - x0)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        batch_size, channels, width, height = image.shape
        assert channels == 1, "Image should have one channel"

        # Generate random control points and sort x-values
        control_points = torch.rand(4, 2, device=image.device, dtype=image.dtype)
        control_points[:, 0], _ = torch.sort(control_points[:, 0])

        t = torch.linspace(-1, 1, steps=512, device=image.device, dtype=image.dtype)
        bezier_map = self._bezier_curve(control_points, t).clamp(-1, 1)

        flattened = image.view(batch_size, -1)
        interp_vals = self._linear_interpolate(flattened, t, bezier_map).view_as(image)

        # Decide whether to invert the transformation based on delta
        transformed = interp_vals if torch.rand(1).item() > self.delta else 1 - interp_vals
        return self._normalize(transformed)


class RealisticElasticDeformation:
    """
    Apply realistic elastic deformation using a multi-level displacement field.
    """

    def __init__(self, max_displacement: int = 100, num_control_points: int = 100,
                 sigma: int = 10, levels: int = 4) -> None:
        self.max_displacement = max_displacement
        self.num_control_points = num_control_points
        self.sigma = sigma
        self.levels = levels

    def _generate_displacement_field(self, h: int, w: int) -> torch.Tensor:
        disp_field = torch.zeros(2, h, w, dtype=torch.float32)
        for level in range(self.levels):
            current_max = self.max_displacement / (2 ** level)
            cp = self.num_control_points
            disp_y = torch.randint(-int(current_max), int(current_max) + 1, (cp, cp), dtype=torch.float32)
            disp_x = torch.randint(-int(current_max), int(current_max) + 1, (cp, cp), dtype=torch.float32)
            # Smooth the displacements
            disp_y = torch.tensor(gaussian_filter(disp_y.numpy(), sigma=self.sigma / (2 ** level)), dtype=torch.float32)
            disp_x = torch.tensor(gaussian_filter(disp_x.numpy(), sigma=self.sigma / (2 ** level)), dtype=torch.float32)
            # Upsample to image resolution
            disp_y = F.interpolate(disp_y.unsqueeze(0).unsqueeze(0), size=(h, w),
                                   mode='bilinear', align_corners=True).squeeze()
            disp_x = F.interpolate(disp_x.unsqueeze(0).unsqueeze(0), size=(h, w),
                                   mode='bilinear', align_corners=True).squeeze()
            disp_field[0] += disp_y
            disp_field[1] += disp_x
        return disp_field

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if not isinstance(image, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")
        if image.dim() != 3:
            raise ValueError("Input tensor must have shape (C, H, W)")
        c, h, w = image.shape
        disp_field = self._generate_displacement_field(h, w)

        y, x = torch.meshgrid(torch.arange(h, device=image.device),
                              torch.arange(w, device=image.device), indexing='ij')
        # Add displacements
        y = y.float() + disp_field[0]
        x = x.float() + disp_field[1]
        # Normalize grid to [-1, 1]
        y = 2.0 * (y / (h - 1)) - 1.0
        x = 2.0 * (x / (w - 1)) - 1.0
        grid = torch.stack((x, y), dim=-1).unsqueeze(0)
        image = image.unsqueeze(0)
        deformed = F.grid_sample(image, grid, align_corners=True, mode='bilinear')
        return deformed.squeeze(0)

    def visualize_dvf(self, disp_field: torch.Tensor, step: int = 8) -> None:
        h, w = disp_field.shape[1:]
        y, x = torch.meshgrid(torch.arange(0, h, step),
                              torch.arange(0, w, step), indexing='ij')
        u = disp_field[0, ::step, ::step].cpu().numpy()
        v = disp_field[1, ::step, ::step].cpu().numpy()
        plt.figure(figsize=(8, 8))
        plt.quiver(x.cpu().numpy(), y.cpu().numpy(), u, v, angles='xy', scale_units='xy', scale=1)
        plt.title("DVF")
        plt.gca().invert_yaxis()
        plt.show()

    def visualize_dvf_grid(self, disp_field: torch.Tensor) -> np.ndarray:
        if disp_field.shape[0] != 2:
            raise ValueError("Displacement field must have shape (2, H, W)")
        _, h, w = disp_field.shape
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        disp_np = disp_field.detach().cpu().numpy()
        x_def = x + disp_np[0, :, :]
        y_def = y + disp_np[1, :, :]
        fig, ax = plt.subplots(figsize=(8, 8))
        for i in range(h):
            ax.plot(x_def[i, :], y_def[i, :], 'k-', linewidth=0.5)
        for j in range(w):
            ax.plot(x_def[:, j], y_def[:, j], 'k-', linewidth=0.5)
        ax.invert_yaxis()
        ax.axis('off')
        ax.set_title("DVF Grid")
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return img
