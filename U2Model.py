import torch
import torch.nn as nn
from Module.base import gaussian_weights_init
from utils.pre_utils import StochasticNonLinearIntensityTransformation
from Loss import Grad, MutualInformation2D
from Module.Network import SpatialTransformer, Encoder, RDP, Decoder
device = "cuda:0"


class MD_multi(nn.Module):
    """
    Multi-Domain model for image translation with feature warping and mutual information loss.
    """

    def __init__(self):
        super(MD_multi, self).__init__()

        # Hyperparameters and components initialization
        self.lr = 0.0001
        self.content_encoder = Encoder(in_channel=1)
        self.decoder = Decoder()
        self.reg = RDP((192, 192))

        # Loss functions
        self.L1 = nn.L1Loss()
        self.grad = Grad()
        self.MI = MutualInformation2D(device=device)
        self.aug = StochasticNonLinearIntensityTransformation(delta=0.5)
        self.STN = SpatialTransformer(size=(192, 192)).to(device)

        # Optimizers
        self.reg_opt = torch.optim.Adam(self.reg.parameters(), lr=self.lr * 2, betas=(0.5, 0.999))
        self.enc_opt = torch.optim.Adam(self.content_encoder.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.dec_opt = torch.optim.Adam(self.decoder.parameters(), lr=self.lr, betas=(0.5, 0.999))

        # Initialize model components and set the device
        self.set_gpu()
        self.initialize_weights()
        self.count_parameters()

    def initialize_weights(self):
        """Apply Gaussian weight initialization to the encoder, decoder, and registration network."""
        self.content_encoder.apply(gaussian_weights_init)
        self.reg.apply(gaussian_weights_init)
        self.decoder.apply(gaussian_weights_init)

    def update_lr(self, lr_decay_factor=0.9):
        """
        Update the learning rate for all optimizers.
        """
        for opt in [self.reg_opt, self.dec_opt, self.enc_opt]:
            for param_group in opt.param_groups:
                param_group['lr'] *= lr_decay_factor  # Decay learning rate
        print(f"Updated learning rate to: {self.reg_opt.param_groups[0]['lr']}")

    def set_gpu(self):
        """Move the model components to GPU."""
        self.content_encoder.to(device)
        self.reg.to(device)
        self.decoder.to(device)

    def forward(self, data):
        """
        Forward pass through the model.
        Takes data, performs transformations, warps, and computes loss.
        """
        self.input = data
        self.real_A = self.input['A'].to(device)
        self.real_B = self.input['B'].to(device)
        self.c_org_A = self.input['A_code'].to(device)
        self.c_org_B = self.input['B_code'].to(device)

        # Apply stochastic intensity transformation (augmentation)
        self.aug_A = self.aug(self.real_A)
        self.aug_B = self.aug(self.real_B)

        # Encode the transformed images
        self.fea_A = self.content_encoder(self.aug_A)
        self.fea_B = self.content_encoder(self.aug_B)

        # Warp the images using the registration network
        self.warp_A2B, self.dvf_a2b = self.reg(self.fea_A, self.fea_B, self.real_A)
        self.dvf_b2a = -self.dvf_a2b
        self.warp_B2A = self.STN(self.real_B, self.dvf_b2a)

        # Re-encode the warped images
        self.warpfea_B2A = self.content_encoder(self.warp_B2A)
        self.warpfea_A2B = self.content_encoder(self.warp_A2B)

        # Decode the features into the fake images
        self.fake_A2B = self.decoder(self.warpfea_A2B, self.c_org_B)
        self.fake_B2A = self.decoder(self.warpfea_B2A, self.c_org_A)

        # Reconstruct images by decoding again
        self.recon_A = self.decoder(self.content_encoder(self.decoder(self.fea_A, self.c_org_B)), self.c_org_A)

        # Compute losses
        self.loss_A2B = self.MI(self.warp_A2B, self.real_B)
        self.loss_B2A = self.MI(self.warp_B2A, self.real_A)
        self.loss_fake_recon = self.L1(self.fake_A2B, self.real_B) + self.L1(self.fake_B2A, self.real_A)
        self.smooth = self.grad.loss(self.dvf_a2b) + self.grad.loss(self.dvf_b2a)
        self.loss_recon = self.L1(self.recon_A, self.real_A)

        # Total generator loss
        self.loss_G = self.smooth + 10 * self.loss_fake_recon + self.loss_recon + self.loss_A2B + self.loss_B2A

    def optimize_parameters(self, data):
        """
        Optimize parameters by computing gradients and stepping through each optimizer.
        """
        self.forward(data)
        self.enc_opt.zero_grad()
        self.reg_opt.zero_grad()
        self.dec_opt.zero_grad()
        self.loss_G.backward()
        self.enc_opt.step()
        self.dec_opt.step()
        self.reg_opt.step()

    def test_reg(self, data):
        """
        Testing phase for registration.
        """
        with torch.no_grad():
            self.input = data
            self.real_A = self.input['A'].to(device)
            self.real_B = self.input['B'].to(device)
            self.fea_A = self.content_encoder(self.real_A)
            self.fea_B = self.content_encoder(self.real_B)

            # Perform warping for registration
            self.warp_A2B, self.dvf_a2b = self.reg(self.fea_A, self.fea_B, self.real_A)
            self.warp_B2A, self.dvf_b2a = self.reg(self.fea_B, self.fea_A, self.real_B)

            # Compute Mutual Information (MI) loss for registration
            self.nmi_A2B = self.MI(self.warp_A2B, self.real_B)
            self.nmi_B2A = self.MI(self.warp_B2A, self.real_A)

    def test_syn(self, data):
        """
        Testing phase for synthetic image generation.
        """
        with torch.no_grad():
            self.input = data
            self.real_A = self.input['A'].to(device)
            self.real_B = self.input['B'].to(device)
            self.c_org_A = self.input['A_code'].to(device)
            self.c_org_B = self.input['B_code'].to(device)

            # Encode and decode the images
            self.fea_A = self.content_encoder(self.real_A)
            self.fea_B = self.content_encoder(self.real_B)
            self.recon_A = self.decoder(self.fea_A, self.c_org_A)
            self.recon_B = self.decoder(self.fea_B, self.c_org_B)

            # Generate fake images using the decoder
            self.fake_A2B = self.decoder(self.fea_A, self.c_org_B)
            self.fake_B2A = self.decoder(self.fea_B, self.c_org_A)

            # Compute PSNR and MAE for synthetic testing
            self.psnr_test = self.PSNR(self.fake_A2B, self.real_B)
            self.mae_test = self.L1(self.fake_A2B, self.real_B)

    def PSNR(self, fake, real):
        """
        Compute the Peak Signal-to-Noise Ratio (PSNR) between fake and real images.
        """
        mse = torch.mean(((fake + 1) - (real + 1)) ** 2)
        if mse < 1.0e-10:
            return 100
        else:
            PIXEL_MAX = 2
            return 10 * torch.log10(PIXEL_MAX / torch.sqrt(mse))

    def count_parameters(self):
        """Print the number of trainable parameters in the model."""
        print("MaFE parameters:",
              sum(p.numel() for p in self.content_encoder.parameters() if p.requires_grad))
        print("MDirT parameters:", sum(p.numel() for p in self.decoder.parameters() if p.requires_grad))
        print("PyReg parameters:", sum(p.numel() for p in self.reg.parameters() if p.requires_grad))


if __name__ == '__main__':
    # Example usage of the model with random data
    model = MD_multi()
    xA = torch.randn(2, 1, 192, 192)
    xB = torch.randn(2, 1, 192, 192)
    cA = torch.randn(2, 3)
    cB = torch.randn(2, 3)

    data_dict = {
        "A": xA,
        "B": xB,
        "A_code": cA,
        "B_code": cB
    }

    model(data_dict)  # Forward pass
    model.optimize_parameters(data_dict)  # Optimize parameters
