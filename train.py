import os
import torch
import pandas as pd
from tqdm import tqdm
import matplotlib
import torchvision.transforms as transforms
from U2Model import MD_multi
from dataloader_BraTs_pair import BraTSDataset, DataLoader
from utils.pre_utils import RealisticElasticDeformation
from utils.train_utils import ImageSaver, plot_metrics

matplotlib.use('agg')

def train_once(model, epoch, train_loader):
    model.train()  # Set model to training mode
    tqdm_iterator = tqdm(train_loader)

    # Initialize accumulators
    total_loss_sum, trans_loss_sum, self_loss_sum = 0, 0, 0
    NMI_A2B_sum, NMI_B2A_sum, smooth_loss_sum = 0, 0, 0
    num_batches = 0

    for i, data in enumerate(tqdm_iterator):
        model.optimize_parameters(data)

        # Accumulate losses
        total_loss_sum += model.loss_G.item()
        self_loss_sum += model.loss_recon.item()
        trans_loss_sum += model.loss_fake_recon.item()
        NMI_A2B_sum += model.loss_A2B.item()
        NMI_B2A_sum += model.loss_B2A.item()
        smooth_loss_sum += model.smooth.item()
        num_batches += 1

        # Update tqdm display
        tqdm_iterator.set_postfix(
            epoch=epoch,
            total_loss=model.loss_G.item(),
            fake_trans=model.loss_fake_recon.item(),
            self_loss=model.loss_recon.item(),
            NMI_A2B=model.loss_A2B.item(),
            NMI_B2A=model.loss_B2A.item(),
            smooth_loss=model.smooth.item(),
            lr1=model.reg_opt.param_groups[0]['lr'],
        )

    # Compute average losses
    avg_losses = {
        'total_loss': total_loss_sum / num_batches,
        'trans_loss': trans_loss_sum / num_batches,
        'self_loss': self_loss_sum / num_batches,
        'NMI_A2B': NMI_A2B_sum / num_batches,
        'NMI_B2A': NMI_B2A_sum / num_batches,
        'smooth_loss': smooth_loss_sum / num_batches,
    }

    return avg_losses

def test_once(model, epoch, test_warp_loader, test_syn_loader, save_dir):
    model.eval()
    img_dir = os.path.join(save_dir, 'img')
    os.makedirs(img_dir, exist_ok=True)

    mae_trans, nmi_A2B, nmi_B2A, psnr_trans = 0, 0, 0, 0
    num_batches = 0

    # Testing with synthetic loader
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

    # Testing with warp loader
    for i, data in enumerate(test_warp_loader):
        model.test_reg(data)
        nmi_A2B += model.nmi_A2B.item()
        psnr_trans += model.psnr_test.item()
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

    # Compute average metrics
    mae_trans /= num_batches
    nmi_A2B /= num_batches
    psnr_trans /= num_batches
    nmi_B2A /= num_batches

    print(f"mae: {mae_trans}, psnr: {psnr_trans}, nmi A2B: {nmi_A2B}, nmi B2A: {nmi_B2A}")
    return mae_trans, psnr_trans, nmi_A2B, nmi_B2A

def train():
    batch_size = 12
    model = MD_multi()
    root_folder = '/home/mpadmin/BraTs/'
    save_dir = "./save_our/"
    os.makedirs(save_dir, exist_ok=True)

    transform = transforms.Compose([RealisticElasticDeformation(max_displacement=120, num_control_points=100, sigma=20)])
    train_dataset = BraTSDataset(root_folder=root_folder, train=True, transform=transform, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    test_warp_dataset = BraTSDataset(root_folder=root_folder, train=False, transform=transform, shuffle=True)
    test_warp_loader = DataLoader(test_warp_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    test_syn_dataset = BraTSDataset(root_folder=root_folder, train=False, transform=None, shuffle=True)
    test_syn_loader = DataLoader(test_syn_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Logging lists
    metrics = {
        'loss_all': [], 'trans_train': [], 'self_recon': [], 'L1_A2B_train': [], 'L1_B2A_train': [], 'smooth_train': [],
        'mae_list': [], 'psnr_list': [], 'nmiA2B_list': [], 'nmiB2A_list': []
    }

    for epoch in range(0, 401):
        avg_losses = train_once(model, epoch, train_loader)
        mae_trans, psnr_trans, nmi_A2B, nmi_B2A = test_once(model, epoch, test_warp_loader, test_syn_loader, save_dir)

        # Append to respective lists
        for key, value in avg_losses.items():
            metrics[key].append(value)

        metrics['mae_list'].append(mae_trans)
        metrics['psnr_list'].append(psnr_trans)
        metrics['nmiA2B_list'].append(nmi_A2B)
        metrics['nmiB2A_list'].append(nmi_B2A)

        # Save metrics to CSV
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

        with open(os.path.join(save_dir, 'metrics_log.csv'), mode='a') as f:
            df.to_csv(f, header=f.tell() == 0, index=False)

        plot_metrics(metrics['loss_all'], metrics['trans_train'], metrics['self_recon'], metrics['L1_A2B_train'], metrics['L1_B2A_train'],
                     metrics['smooth_train'], metrics['mae_list'], metrics['psnr_list'], metrics['nmiA2B_list'], metrics['nmiB2A_list'],
                     os.path.join(save_dir, 'plt'))

        # Save model every 5 epochs
        if epoch % 5 == 0:
            torch.save(model.content_encoder.state_dict(), os.path.join(save_dir, f'{epoch}_mafe.pth'))
            torch.save(model.decoder.state_dict(), os.path.join(save_dir, f'{epoch}_mdirt.pth'))
            torch.save(model.reg.state_dict(), os.path.join(save_dir, f'{epoch}_reg.pth'))

        # Update learning rate every 10 epochs
        if epoch > 0 and epoch % 10 == 0:
            model.update_lr()

if __name__ == '__main__':
    train()
