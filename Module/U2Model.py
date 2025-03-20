import torch
import torch.nn as nn
from .Base import gaussian_weights_init
from utils.pre_utils import StochasticNonLinearIntensityTransformation
from Loss import Grad, MutualInformation2D
from .Network import SpatialTransformer, MDirT, MaFE, PyReg

class U2Model(nn.Module):
    def __init__(self, opt):
        super(U2Model, self).__init__()
        self.opt = opt
        self.device = opt.device
        self.lr = opt.learning_rate

        self.build_modules()
        self.configure_optimizers()
        self.to(self.device)
        self.initialize_weights()
        self.count_parameters()

    def build_modules(self):
        # Build submodules using parameters from opt
        self.content_encoder = MaFE(in_channel=self.opt.in_channel)
        self.decoder = MDirT()
        self.reg = PyReg(inshape=self.opt.image_shape,
                       in_channel=self.opt.in_channel,
                       channels=self.opt.channels)
        self.L1 = nn.L1Loss()
        self.grad = Grad()
        self.MI = MutualInformation2D(device=self.device)
        self.aug = StochasticNonLinearIntensityTransformation(delta=self.opt.delta)
        self.STN = SpatialTransformer(size=self.opt.image_shape).to(self.device)

    def configure_optimizers(self):
        # Configure optimizers
        self.reg_opt = torch.optim.Adam(self.reg.parameters(), lr=self.lr * 2, betas=(0.5, 0.999))
        self.enc_opt = torch.optim.Adam(self.content_encoder.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.dec_opt = torch.optim.Adam(self.decoder.parameters(), lr=self.lr, betas=(0.5, 0.999))

    def initialize_weights(self):
        # Apply Gaussian weight initialization
        self.content_encoder.apply(gaussian_weights_init)
        self.reg.apply(gaussian_weights_init)
        self.decoder.apply(gaussian_weights_init)

    def compute_losses(self, real_A, real_B, c_org_A, c_org_B, fea_A, fea_B):
        # Compute all intermediate values and store them as self.xxx
        self.warp_A2B, self.dvf_a2b = self.reg(fea_A, fea_B, real_A)
        self.dvf_b2a = -self.dvf_a2b
        self.warp_B2A = self.STN(real_B, self.dvf_b2a)
        self.warpfea_A2B = self.content_encoder(self.warp_A2B)
        self.warpfea_B2A = self.content_encoder(self.warp_B2A)
        self.fake_A2B = self.decoder(self.warpfea_A2B, c_org_B)
        self.fake_B2A = self.decoder(self.warpfea_B2A, c_org_A)
        self.recon_A = self.decoder(self.content_encoder(self.decoder(fea_A, c_org_B)), c_org_A)
        self.recon_B = self.decoder(self.content_encoder(self.decoder(fea_B, c_org_A)), c_org_B)
        self.loss_A2B = self.MI(self.warp_A2B, real_B)
        self.loss_B2A = self.MI(self.warp_B2A, real_A)
        self.loss_fake_recon = self.L1(self.fake_A2B, real_B) + self.L1(self.fake_B2A, real_A)
        self.smooth = self.grad.loss(self.dvf_a2b) + self.grad.loss(self.dvf_b2a)
        self.loss_recon = self.L1(self.recon_A, real_A)+self.L1(self.recon_B, real_B)

        self.loss_G = (self.opt.weight_smooth * self.smooth +
                       self.opt.weight_fake_recon * self.loss_fake_recon +
                       self.opt.weight_recon * self.loss_recon +
                       self.opt.weight_nmi * (self.loss_A2B +self.loss_B2A))
        return self.loss_G

    def forward(self, data):
        # Move inputs to device and store as self.xxx
        self.real_A = data['A'].to(self.device)
        self.real_B = data['B'].to(self.device)
        self.c_org_A = data['A_code'].to(self.device)
        self.c_org_B = data['B_code'].to(self.device)

        self.aug_A = self.aug(self.real_A)
        self.aug_B = self.aug(self.real_B)

        self.fea_A = self.content_encoder(self.aug_A)
        self.fea_B = self.content_encoder(self.aug_B)

        loss_G = self.compute_losses(self.real_A, self.real_B, self.c_org_A, self.c_org_B, self.fea_A, self.fea_B)
        return loss_G

    def optimize_parameters(self, data):
        loss_G = self.forward(data)
        self.enc_opt.zero_grad()
        self.reg_opt.zero_grad()
        self.dec_opt.zero_grad()
        loss_G.backward()
        self.enc_opt.step()
        self.dec_opt.step()
        self.reg_opt.step()

    def test_reg(self, data):
        with torch.no_grad():
            self.real_A = data['A'].to(self.device)
            self.real_B = data['B'].to(self.device)
            self.fea_A = self.content_encoder(self.real_A)
            self.fea_B = self.content_encoder(self.real_B)
            self.warp_A2B, self.dvf_a2b = self.reg(self.fea_A, self.fea_B, self.real_A)
            self.warp_B2A, self.dvf_b2a = self.reg(self.fea_B, self.fea_A, self.real_B)
            self.nmi_A2B = self.MI(self.warp_A2B, self.real_B)
            self.nmi_B2A = self.MI(self.warp_B2A, self.real_A)

    def test_syn(self, data):
        with torch.no_grad():
            self.real_A = data['A'].to(self.device)
            self.real_B = data['B'].to(self.device)
            self.c_org_A = data['A_code'].to(self.device)
            self.c_org_B = data['B_code'].to(self.device)
            self.fea_A = self.content_encoder(self.real_A)
            self.fea_B = self.content_encoder(self.real_B)
            self.recon_A = self.decoder(self.fea_A, self.c_org_A)
            self.recon_B = self.decoder(self.fea_B, self.c_org_B)
            self.fake_A2B = self.decoder(self.fea_A, self.c_org_B)
            self.fake_B2A = self.decoder(self.fea_B, self.c_org_A)
            self.psnr_test = self.PSNR(self.fake_A2B, self.real_B)
            self.mae_test = self.L1(self.fake_A2B, self.real_B)

    def PSNR(self, fake, real):
        mse = torch.mean(((fake + 1) - (real + 1)) ** 2)
        if mse < 1.0e-10:
            return 100
        PIXEL_MAX = 2
        return 10 * torch.log10(PIXEL_MAX / torch.sqrt(mse))

    def count_parameters(self):
        print("Content Encoder parameters:",
              sum(p.numel() for p in self.content_encoder.parameters() if p.requires_grad))
        print("Decoder parameters:",
              sum(p.numel() for p in self.decoder.parameters() if p.requires_grad))
