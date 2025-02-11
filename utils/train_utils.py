import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage.metrics import structural_similarity as ssim
import math
from utils.pre_utils import RealisticElasticDeformation
class ModelSaver:
    """Handles model saving to specified directory."""
    @staticmethod
    def save_models(save_dir, model, epoch):
        os.makedirs(save_dir, exist_ok=True)
        # torch.save(model.content_encoder.state_dict(), os.path.join(save_dir, f'{epoch}_Cont_E.pth'))
        torch.save(model.reg.state_dict(), os.path.join(save_dir, f'{epoch}_RegNet.pth'))
        # torch.save(model.decoder.state_dict(), os.path.join(save_dir, f'{epoch}_G.pth'))


class Plotter:
    """Handles loss plotting to visualize training and testing metrics."""

    @staticmethod
    def plot_metrics(list_all, list_A2B_mi_train, list_B2A_mi_train, list_smooth,
                     list_mi_A2B, list_mi_B2A,
                     save_dir):
        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)

        epochs = range(len(list_all))  # Dynamically use the length of list_all for the x-axis

        # Plot Total Loss (training)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, list_all, label="Total Loss", color="purple")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Total Loss over Epochs")
        plt.legend()
        plt.savefig(os.path.join(save_dir, "total_loss.png"))
        plt.close()

        # Plot L1 Loss (train vs. test)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, list_A2B_mi_train, label="A2B MI Loss (Train)", color="blue")
        plt.plot(epochs, list_B2A_mi_train, label="B2A MI Loss (Train)", color="yellow")
        plt.plot(epochs, list_mi_A2B, label="A2B MI Loss (Test)", color="blue", linestyle="--")
        plt.plot(epochs, list_mi_B2A, label="B2A MI Loss (Test)", color="yellow", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("MI Loss")
        plt.title("MI Loss over Epochs (Train vs. Test)")
        plt.legend()
        plt.savefig(os.path.join(save_dir, "MI_loss.png"))
        plt.close()

        # Plot PSNR (train vs. test)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, list_smooth, label="Smooth", color="Orange")
        plt.xlabel("Epoch")
        plt.ylabel("PSNR")
        plt.title("Smooth over Epochs")
        plt.legend()
        plt.savefig(os.path.join(save_dir, "smooth.png"))
        plt.close()


class ImageSaver:
    @staticmethod
    def save_images_to_directory(epoch, save_dir, real_A, real_B, warp_A2B,warp_B2A, DVF_A2B, DVF_B2A):
        """
        Save a 2x3 grid of images (real, self, and recon) to a specified directory.

        Args:
        - epoch: The current epoch (used to name the saved image).
        - save_dir: The directory to save the images.
        - real_A, real_B, self_A, self_B, recon_B2A, recon_A2B: The images to be plotted and saved.
        """
        # Ensure the save directory exists
        # 调用 visualize_dvf_grid 方法
        os.makedirs(save_dir, exist_ok=True)

        # Create a 2x3 subplot
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))

        # Plot real A and real B in the first column
        axes[0, 0].imshow(real_A, cmap='gray')
        axes[0, 0].set_title("Real A")
        axes[0, 0].axis('off')

        axes[1, 0].imshow(real_B, cmap='gray')
        axes[1, 0].set_title("Real B")
        axes[1, 0].axis('off')

        # Plot self A and self B in the second column
        axes[0, 1].imshow(warp_A2B, cmap='gray')
        axes[0, 1].set_title("warp_A2B")
        axes[0, 1].axis('off')

        axes[1, 1].imshow(warp_B2A, cmap='gray')
        axes[1, 1].set_title("warp_B2A")
        axes[1, 1].axis('off')

        # Plot recon B2A and recon A2B in the third column
        axes[0, 2].imshow(flow2rgb(DVF_A2B))
        axes[0, 2].set_title("DVF_A2B")
        axes[0, 2].axis('off')

        axes[1, 2].imshow(flow2rgb(DVF_B2A))
        axes[1, 2].set_title("DVF_B2A")
        axes[1, 2].axis('off')

        # Save the figure to the specified directory
        image_path = os.path.join(save_dir, f"epoch_{epoch}_images.png")
        plt.tight_layout()
        plt.savefig(image_path)
        plt.close()  # Close the plot to avoid memory issues

        print(f"Images saved to {image_path}")

    def save_reg_image(epoch, save_dir, real_A, real_B, warp_A2B, warp_B2A, DVF_A2B, DVF_B2A):
        """
        Save a 2x3 grid of registration-related images (real, warp, DVF) to a specified directory.

        Args:
        - epoch: The current epoch (used to name the saved image).
        - save_dir: The directory to save the images.
        - real_A, real_B: Input images.
        - warp_A2B, warp_B2A: Warped images.
        - DVF_A2B, DVF_B2A: Displacement vector fields.
        """
        os.makedirs(save_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))

        axes[0, 0].imshow(real_A, cmap='gray')
        axes[0, 0].set_title("Real A")
        axes[0, 0].axis('off')

        axes[1, 0].imshow(real_B, cmap='gray')
        axes[1, 0].set_title("Real B")
        axes[1, 0].axis('off')

        axes[0, 1].imshow(warp_A2B, cmap='gray')
        axes[0, 1].set_title("Warp A2B")
        axes[0, 1].axis('off')

        axes[1, 1].imshow(warp_B2A, cmap='gray')
        axes[1, 1].set_title("Warp B2A")
        axes[1, 1].axis('off')

        axes[0, 2].imshow(flow2rgb(DVF_A2B))
        axes[0, 2].set_title("DVF A2B")
        axes[0, 2].axis('off')

        axes[1, 2].imshow(flow2rgb(DVF_B2A))
        axes[1, 2].set_title("DVF B2A")
        axes[1, 2].axis('off')

        image_path = os.path.join(save_dir, f"epoch_{epoch}_reg_images.png")
        plt.tight_layout()
        plt.savefig(image_path)
        plt.close()

        print(f"Registration images saved to {image_path}")

    def save_syn_image(epoch, save_dir, real_A, real_B, fake_A2B, fake_B2A, recon_A, recon_B):
        """
        Save a 2x3 grid of synthesis-related images (real, fake, recon) to a specified directory.

        Args:
        - epoch: The current epoch (used to name the saved image).
        - save_dir: The directory to save the images.
        - real_A, real_B: Input images.
        - fake_A2B, fake_B2A: Synthesized images.
        - recon_A, recon_B: Reconstructed images.
        """
        os.makedirs(save_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))

        axes[0, 0].imshow(real_A, cmap='gray')
        axes[0, 0].set_title("Real A")
        axes[0, 0].axis('off')

        axes[1, 0].imshow(real_B, cmap='gray')
        axes[1, 0].set_title("Real B")
        axes[1, 0].axis('off')

        axes[0, 1].imshow(fake_A2B, cmap='gray')
        axes[0, 1].set_title("Fake A2B")
        axes[0, 1].axis('off')

        axes[1, 1].imshow(fake_B2A, cmap='gray')
        axes[1, 1].set_title("Fake B2A")
        axes[1, 1].axis('off')

        axes[0, 2].imshow(recon_A, cmap='gray')
        axes[0, 2].set_title("Recon A")
        axes[0, 2].axis('off')

        axes[1, 2].imshow(recon_B, cmap='gray')
        axes[1, 2].set_title("Recon B")
        axes[1, 2].axis('off')

        image_path = os.path.join(save_dir, f"epoch_{epoch}_syn_images.png")
        plt.tight_layout()
        plt.savefig(image_path)
        plt.close()

        print(f"Synthesis images saved to {image_path}")


def flow2rgb(flow_map):
    """Converts a flow map to an RGB image representation."""
    if isinstance(flow_map, torch.Tensor):
        flow_map_np = flow_map.detach().cpu().numpy()
    else:
        flow_map_np = flow_map

    _, h, w = flow_map_np.shape
    flow_map_np[:, (flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float('nan')
    rgb_map = np.ones((3, h, w)).astype(np.float32)

    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max() + 1e-5)

    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5 * (normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]

    rgb_map = rgb_map.clip(0, 1)
    return np.transpose(rgb_map, (1, 2, 0))




def visualize_dvf_grid(dvf):

    assert dvf.shape[0] == 2, "输入的 dvf 应该是 (2, h, w) 的形状"

    # 获取形变场的形状
    _, h, w = dvf.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # 应用 DVF 到网格
    dvf_np = dvf
    x_deformed = x + dvf_np[0, :, :]
    y_deformed = y + dvf_np[1, :, :]

    # 创建图像并保存为 numpy 数组
    fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(h):
        ax.plot(x_deformed[i, :], y_deformed[i, :], 'k-', linewidth=0.5)
    for j in range(w):
        ax.plot(x_deformed[:, j], y_deformed[:, j], 'k-', linewidth=0.5)

    ax.invert_yaxis()
    ax.axis('off')
    ax.set_title("DVF Grid Visualization")

    # 将绘图转换为 numpy 数组
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # 关闭绘图，释放内存
    plt.close(fig)

    return img

def plot_metrics(avg_total_loss, avg_trans_loss, avg_self_loss, avg_NMI_A2B, avg_NMI_B2A, avg_smooth_loss,
                 mae_trans, psnr_trans, nmi_A2B, nmi_B2A, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.plot(avg_total_loss, label='Total Loss', color='blue')
    plt.title('Total Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, 'total_loss.png'))
    plt.close()

    # 2. NMI (A2B and B2A) - Train and Test
    plt.figure(figsize=(8, 6))
    plt.plot(avg_NMI_A2B, label='NMI A2B - Train', color='blue')
    plt.plot(avg_NMI_B2A, label='NMI B2A - Train', color='green')
    plt.plot(nmi_A2B, label='NMI A2B - Test', color='blue', linestyle='--')
    plt.plot(nmi_B2A, label='NMI B2A - Test', color='green', linestyle='--')
    plt.title('NMI Over Epochs (Train and Test)')
    plt.xlabel('Epochs')
    plt.ylabel('NMI')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, 'nmi_metrics.png'))
    plt.close()

    # 3. PSNR
    plt.figure(figsize=(8, 6))
    plt.plot(psnr_trans, label='PSNR', color='purple')
    plt.title('PSNR Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, 'psnr.png'))
    plt.close()

    # 4. Smooth Loss
    plt.figure(figsize=(8, 6))
    plt.plot(avg_smooth_loss, label='Smooth Loss', color='teal')
    plt.title('Smooth Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, 'smooth_loss.png'))
    plt.close()

    # 5. Trans Loss, Self Loss, and MAE
    plt.figure(figsize=(8, 6))
    plt.plot(avg_trans_loss, label='Trans Loss', color='blue')
    plt.plot(avg_self_loss, label='Self Loss', color='green')
    plt.plot(mae_trans, label='MAE', color='red')
    plt.title('Trans Loss, Self Loss, and MAE Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, 'trans_self_mae.png'))
    plt.close()