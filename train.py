import os
import torch
import pandas as pd
from tqdm import tqdm
import matplotlib
import torchvision.transforms as transforms
import logging

from Module.U2Model import U2Model
from dataloader_BraTs_pair import BraTSDataset, DataLoader
from utils.pre_utils import RealisticElasticDeformation
from utils.train_utils import ImageSaver, plot_metrics

matplotlib.use('agg')


def setup_logger(save_dir):
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)
    log_file = os.path.join(save_dir, "train.log")
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


def train_once(model, epoch, train_loader, logger):
    model.train()
    tqdm_iterator = tqdm(train_loader)
    total_loss_sum, trans_loss_sum, self_loss_sum = 0, 0, 0
    NMI_A2B_sum, NMI_B2A_sum, smooth_loss_sum = 0, 0, 0
    num_batches = 0

    for i, data in enumerate(tqdm_iterator):
        try:
            model.optimize_parameters(data)
        except Exception as e:
            logger.error(f"Epoch {epoch} Batch {i}: Optimization error: {e}")
            continue

        total_loss_sum += model.loss_G.item()
        self_loss_sum += model.loss_recon.item()
        trans_loss_sum += model.loss_fake_recon.item()
        NMI_A2B_sum += model.loss_A2B.item()
        NMI_B2A_sum += model.loss_B2A.item()
        smooth_loss_sum += model.smooth.item()
        num_batches += 1

        tqdm_iterator.set_postfix(
            epoch=epoch,
            total_loss=model.loss_G.item(),
            fake_trans=model.loss_fake_recon.item(),
            self_loss=model.loss_recon.item(),
            NMI_A2B=model.loss_A2B.item(),
            NMI_B2A=model.loss_B2A.item(),
            smooth_loss=model.smooth.item(),
            lr=model.reg_opt.param_groups[0]['lr']
        )

        logger.info(f"Epoch {epoch} Batch {i}: Total Loss: {model.loss_G.item():.4f}, "
                    f"Self Loss: {model.loss_recon.item():.4f}, Fake Trans Loss: {model.loss_fake_recon.item():.4f}, "
                    f"NMI A2B: {model.loss_A2B.item():.4f}, NMI B2A: {model.loss_B2A.item():.4f}, "
                    f"Smooth: {model.smooth.item():.4f}, LR: {model.reg_opt.param_groups[0]['lr']:.6f}")

    avg_losses = {
        'total_loss': total_loss_sum / num_batches if num_batches else 0,
        'trans_loss': trans_loss_sum / num_batches if num_batches else 0,
        'self_loss': self_loss_sum / num_batches if num_batches else 0,
        'NMI_A2B': NMI_A2B_sum / num_batches if num_batches else 0,
        'NMI_B2A': NMI_B2A_sum / num_batches if num_batches else 0,
        'smooth_loss': smooth_loss_sum / num_batches if num_batches else 0,
    }
    return avg_losses


def test_once(model, epoch, test_warp_loader, test_syn_loader, save_dir, logger):
    model.eval()
    img_dir = os.path.join(save_dir, 'img')
    os.makedirs(img_dir, exist_ok=True)
    mae_trans, nmi_A2B, nmi_B2A, psnr_trans = 0, 0, 0, 0
    num_batches = 0

    for i, data in enumerate(test_syn_loader):
        model.test_syn(data)
        mae_trans += model.mae_test.item()
        psnr_trans += model.psnr_test.item()
        if i == 0:
            ImageSaver.save_syn_image(epoch, img_dir,
                model.real_A.cpu().numpy().squeeze(1)[0],
                model.real_B.cpu().numpy().squeeze(1)[0],
                model.fake_A2B.cpu().numpy().squeeze(1)[0],
                model.fake_B2A.cpu().numpy().squeeze(1)[0],
                model.recon_A.cpu().numpy().squeeze(1)[0],
                model.recon_B.cpu().numpy().squeeze(1)[0]
            )
        logger.info(f"Test Synthetic Epoch {epoch} Batch {i}: MAE: {model.mae_test.item():.4f}, PSNR: {model.psnr_test.item():.4f}")

    for i, data in enumerate(test_warp_loader):
        model.test_reg(data)
        nmi_A2B += model.nmi_A2B.item()
        nmi_B2A += model.nmi_B2A.item()
        num_batches += 1
        if i == 0:
            ImageSaver.save_reg_image(epoch, img_dir,
                model.real_A.cpu().numpy().squeeze(1)[0],
                model.real_B.cpu().numpy().squeeze(1)[0],
                model.warp_A2B.cpu().numpy().squeeze(1)[0],
                model.warp_B2A.cpu().numpy().squeeze(1)[0],
                model.dvf_a2b.cpu().numpy()[0],
                model.dvf_b2a.cpu().numpy()[0]
            )
        logger.info(f"Test Registration Epoch {epoch} Batch {i}: NMI A2B: {model.nmi_A2B.item():.4f}, NMI B2A: {model.nmi_B2A.item():.4f}")

    if num_batches > 0:
        mae_trans /= num_batches
        nmi_A2B /= num_batches
        psnr_trans /= num_batches
        nmi_B2A /= num_batches
    logger.info(f"Epoch {epoch} Test Metrics: MAE: {mae_trans:.4f}, PSNR: {psnr_trans:.4f}, "
                f"NMI A2B: {nmi_A2B:.4f}, NMI B2A: {nmi_B2A:.4f}")
    return mae_trans, psnr_trans, nmi_A2B, nmi_B2A


def train(opt):
    os.makedirs(opt.save_dir, exist_ok=True)
    logger = setup_logger(opt.save_dir)
    logger.info("Starting training process...")

    # Initialize model with options
    model = U2Model(opt)
    model.to(model.device)

    root_folder = opt.root_folder
    transform = transforms.Compose([
        RealisticElasticDeformation(max_displacement=opt.max_displacement,
                                    num_control_points=opt.num_control_points,
                                    sigma=opt.sigma)
    ])
    train_dataset = BraTSDataset(root_folder=root_folder, train=True, transform=transform, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    test_warp_dataset = BraTSDataset(root_folder=root_folder, train=False, transform=transform, shuffle=True)
    test_warp_loader = DataLoader(test_warp_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    test_syn_dataset = BraTSDataset(root_folder=root_folder, train=False, transform=None, shuffle=True)
    test_syn_loader = DataLoader(test_syn_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    # Initialize metrics log lists
    metrics = {
        'loss_all': [], 'trans_train': [], 'self_recon': [],
        'L1_A2B_train': [], 'L1_B2A_train': [], 'smooth_train': [],
        'mae_list': [], 'psnr_list': [], 'nmiA2B_list': [], 'nmiB2A_list': []
    }

    for epoch in range(opt.epochs):
        logger.info(f"Epoch {epoch} start.")
        avg_losses = train_once(model, epoch, train_loader, logger)
        mae_trans, psnr_trans, nmi_A2B, nmi_B2A = test_once(model, epoch, test_warp_loader, test_syn_loader, save_dir, logger)

        for key, value in avg_losses.items():
            metrics[key].append(value)
        metrics['mae_list'].append(mae_trans)
        metrics['psnr_list'].append(psnr_trans)
        metrics['nmiA2B_list'].append(nmi_A2B)
        metrics['nmiB2A_list'].append(nmi_B2A)

        # Save current metrics to CSV
        df = pd.DataFrame({
            'epoch': [epoch],
            'train_total_loss': [avg_losses['total_loss']],
            'train_trans_loss': [avg_losses['trans_loss']],
            'train_self_loss': [avg_losses['self_loss']],
            'train_NMI_A2B': [avg_losses['NMI_A2B']],
            'train_NMI_B2A': [avg_losses['NMI_B2A']],
            'train_smooth_loss': [avg_losses['smooth_loss']],
            'test_mae': [mae_trans],
            'test_psnr': [psnr_trans],
            'test_nmi_A2B': [nmi_A2B],
            'test_nmi_B2A': [nmi_B2A],
        })
        with open(os.path.join(opt.save_dir, 'metrics_log.csv'), mode='a') as f:
            df.to_csv(f, header=f.tell() == 0, index=False)

        # Plot metrics and save figures
        plot_metrics(metrics['loss_all'], metrics['trans_train'], metrics['self_recon'],
                     metrics['L1_A2B_train'], metrics['L1_B2A_train'], metrics['smooth_train'],
                     metrics['mae_list'], metrics['psnr_list'], metrics['nmiA2B_list'], metrics['nmiB2A_list'],
                     os.path.join(opt.save_dir, 'plt'))

        # Save model snapshots every opt.model_save_interval epochs
        if epoch % opt.model_save_interval == 0:
            torch.save(model.content_encoder.state_dict(), os.path.join(opt.save_dir, f'{epoch}_mafe.pth'))
            torch.save(model.decoder.state_dict(), os.path.join(opt.save_dir, f'{epoch}_mdirt.pth'))
            torch.save(model.reg.state_dict(), os.path.join(opt.save_dir, f'{epoch}_reg.pth'))
            logger.info(f"Model snapshots saved at epoch {epoch}.")

        # Update learning rate every opt.lr_update_interval epochs
        if epoch > 0 and epoch % opt.lr_update_interval == 0:
            model.update_lr()
            logger.info(f"Learning rate updated at epoch {epoch}.")

    logger.info("Training completed.")


if __name__ == '__main__':
    from options import get_options
    opt = get_options()  # 解析命令行参数获取 opt
    train(opt)
