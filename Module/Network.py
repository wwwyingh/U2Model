import torch
import torch.nn as nn
from .Base import ConvInsBlock, ResBlock, UpConvBlock, VecInt, CConv, SpatialTransformer, FiLMModulation_drop as FiLMModulation
import options

class MaFE(nn.Module):
    def __init__(self, in_channel=options.in_channel, first_out_channel=options.first_out_channel):
        super(MaFE, self).__init__()
        c = first_out_channel
        self.conv0_1 = ConvInsBlock(in_channel, c, kernel_size=5, stride=1, padding=2)
        self.conv0_2 = ConvInsBlock(c, c * 8, kernel_size=5, stride=1, padding=2)
        self.conv0_3 = ConvInsBlock(c * 8, c * 4, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(c * 4, 2 * c, kernel_size=3, stride=2, padding=1),
            ResBlock(2 * c)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(2 * c, 4 * c, kernel_size=3, stride=2, padding=1),
            ResBlock(4 * c)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(4 * c, 8 * c, kernel_size=3, stride=2, padding=1),
            ResBlock(8 * c)
        )
    def forward(self, x):
        out0 = self.conv0_1(x)
        out02 = self.conv0_2(out0)
        out03 = self.conv0_3(out02)
        out1 = self.conv1(out03)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        return [out0, out1, out2, out3]

class MDirT(nn.Module):
    def __init__(self, out_channel=options.in_channel, first_out_channel=options.first_out_channel, dropout_prob=options.dropout_prob):
        super(MDirT, self).__init__()
        c = first_out_channel
        self.upconv3 = UpConvBlock(8 * c, 4 * c, kernel_size=4, stride=2, alpha=0.1)
        self.conv3 = ConvInsBlock(8 * c, 4 * c, kernel_size=3, stride=1, padding=1)
        self.res3 = ResBlock(4 * c)
        self.dropout3 = nn.Dropout2d(dropout_prob)
        self.upconv2 = UpConvBlock(4 * c, 2 * c, kernel_size=4, stride=2, alpha=0.1)
        self.conv2 = ConvInsBlock(4 * c, 2 * c, kernel_size=3, stride=1, padding=1)
        self.res2 = ResBlock(2 * c)
        self.dropout2 = nn.Dropout2d(dropout_prob)
        self.upconv1 = UpConvBlock(2 * c, c, kernel_size=4, stride=2, alpha=0.1)
        self.conv1 = ConvInsBlock(2 * c, c, kernel_size=3, stride=1, padding=1)
        self.res1 = ResBlock(c)
        self.dropout1 = nn.Dropout2d(dropout_prob)
        self.conv0 = nn.Conv2d(c, out_channel, kernel_size=1, stride=1, padding=0)
        self.act = nn.Tanh()
        self.Film0 = FiLMModulation(input_dim=3, output_dim=8 * c)
        self.Film1 = FiLMModulation(input_dim=3, output_dim=4 * c)
        self.Film2 = FiLMModulation(input_dim=3, output_dim=2 * c)
    def forward(self, encoder_outputs, c_org):
        out0, out1, out2, out3 = encoder_outputs
        y1 = self.Film0(out3, c_org)
        up3 = self.upconv3(y1)
        up3 = self.Film1(up3, c_org)
        concat3 = torch.cat([up3, out2], dim=1)
        conv3 = self.conv3(concat3)
        res3 = self.res3(conv3)
        res3 = self.dropout3(res3)
        up2 = self.upconv2(res3)
        up2 = self.Film2(up2, c_org)
        concat2 = torch.cat([up2, out1], dim=1)
        conv2 = self.conv2(concat2)
        res2 = self.res2(conv2)
        res2 = self.dropout2(res2)
        up1 = self.upconv1(res2)
        concat1 = torch.cat([up1, out0], dim=1)
        conv1 = self.conv1(concat1)
        res1 = self.res1(conv1)
        res1 = self.dropout1(res1)
        out = self.act(self.conv0(res1))
        return out

class PyReg(nn.Module):
    def __init__(self, inshape=options.image_shape, flow_multiplier=options.flow_multiplier, in_channel=options.in_channel, channels=options.channels):
        super(PyReg, self).__init__()
        self.flow_multiplier = flow_multiplier
        self.channels = channels
        self.step = options.registration_steps
        self.inshape = inshape
        c = self.channels
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_bilin = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        en_c = 64
        self.warp = nn.ModuleList()
        self.diff = nn.ModuleList()
        self.conv_0 = ConvInsBlock(en_c, 16, kernel_size=3, stride=1)
        self.conv_1 = ConvInsBlock(2 * en_c, 32, kernel_size=3, stride=1)
        self.conv_2 = ConvInsBlock(4 * en_c, 64, kernel_size=3, stride=1)
        self.conv_3 = ConvInsBlock(8 * en_c, 128, kernel_size=3, stride=1)
        for i in range(4):
            self.warp.append(SpatialTransformer([s // (2 ** i) for s in inshape]))
            self.diff.append(VecInt([s // (2 ** i) for s in inshape]))
        self.cconv_4 = nn.Sequential(
            ConvInsBlock(16 * c, 8 * c, kernel_size=3, stride=1),
            ConvInsBlock(8 * c, 8 * c, kernel_size=3, stride=1)
        )
        self.defconv4 = nn.Conv2d(8 * c, 2, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(self.defconv4.weight, 0, 1e-5)
        nn.init.constant_(self.defconv4.bias, 0)
        self.dconv4 = nn.Sequential(
            ConvInsBlock(3 * 8 * c, 8 * c),
            ConvInsBlock(8 * c, 8 * c)
        )
        self.upconv3 = UpConvBlock(8 * c, 4 * c, kernel_size=4, stride=2)
        self.cconv_3 = CConv(3 * 4 * c)
        self.defconv3 = nn.Conv2d(3 * 4 * c, 2, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(self.defconv3.weight, 0, 1e-5)
        nn.init.constant_(self.defconv3.bias, 0)
        self.dconv3 = ConvInsBlock(3 * 4 * c, 4 * c)
        self.upconv2 = UpConvBlock(3 * 4 * c, 2 * c, kernel_size=4, stride=2)
        self.cconv_2 = CConv(3 * 2 * c)
        self.defconv2 = nn.Conv2d(3 * 2 * c, 2, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(self.defconv2.weight, 0, 1e-5)
        nn.init.constant_(self.defconv2.bias, 0)
        self.dconv2 = ConvInsBlock(3 * 2 * c, 2 * c)
        self.upconv1 = UpConvBlock(3 * 2 * c, c, kernel_size=4, stride=2)
        self.cconv_1 = CConv(3 * c)
        self.defconv1 = nn.Conv2d(3 * c, 2, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(self.defconv1.weight, 0, 1e-5)
        nn.init.constant_(self.defconv1.bias, 0)
    def forward(self, moving_outputs, fixed_outputs, moving):
        M1, M2, M3, M4 = moving_outputs
        F1, F2, F3, F4 = fixed_outputs
        M1 = self.conv_0(M1)
        M2 = self.conv_1(M2)
        M3 = self.conv_2(M3)
        M4 = self.conv_3(M4)
        F1 = self.conv_0(F1)
        F2 = self.conv_1(F2)
        F3 = self.conv_2(F3)
        F4 = self.conv_3(F4)
        C4 = torch.cat([F4, M4], dim=1)
        C4 = self.cconv_4(C4)
        flow = self.defconv4(C4)
        flow = self.diff[3](flow)
        warped = self.warp[3](M4, flow)
        warped_4 = warped
        C4 = self.dconv4(torch.cat([F4, warped, C4], dim=1))
        v = self.defconv4(C4)
        w = self.diff[3](v)
        D3 = self.upconv3(C4)
        flow = self.upsample_bilin(2 * (self.warp[3](flow, w) + w))
        warped = self.warp[2](M3, flow)
        C3 = self.cconv_3(F3, warped, D3)
        v = self.defconv3(C3)
        w = self.diff[2](v)
        flow = self.warp[2](flow, w) + w
        warped = self.warp[2](M3, flow)
        D3 = self.dconv3(C3)
        C3 = self.cconv_3(F3, warped, D3)
        v = self.defconv3(C3)
        w = self.diff[2](v)
        warped_3 = warped
        D2 = self.upconv2(C3)
        flow = self.upsample_bilin(2 * (self.warp[2](flow, w) + w))
        warped = self.warp[1](M2, flow)
        C2 = self.cconv_2(F2, warped, D2)
        v = self.defconv2(C2)
        w = self.diff[1](v)
        flow = self.warp[1](flow, w) + w
        warped = self.warp[1](M2, flow)
        D2 = self.dconv2(C2)
        C2 = self.cconv_2(F2, warped, D2)
        v = self.defconv2(C2)
        w = self.diff[1](v)
        warped_2 = warped
        D1 = self.upconv1(C2)
        flow = self.upsample_bilin(2 * (self.warp[1](flow, w) + w))
        warped = self.warp[0](M1, flow)
        C1 = self.cconv_1(F1, warped, D1)
        v = self.defconv1(C1)
        w = self.diff[0](v)
        flow = self.warp[0](flow, w) + w
        warped = self.warp[0](M1, flow)
        return warped, flow
