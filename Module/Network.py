import torch
import torch.nn as nn
from Module.base_modules import ConvInsBlock, ResBlock, UpConvBlock, VecInt, CConv, Normal, SpatialTransformer,FiLMModulation_drop as FiLMModulation


class Encoder(nn.Module):
    """
    Encoder for 2D images
    """

    def __init__(self, in_channel=1, first_out_channel=64):
        super(Encoder, self).__init__()

        c = first_out_channel
        self.conv0_1 = ConvInsBlock(in_channel, c, 5, 1, padding=2)
        self.conv0_2 = ConvInsBlock(c, c * 8,5, 1, padding=2)
        self.conv0_3 = ConvInsBlock(c * 8, c * 4, 3, 1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(c*4, 2 * c, kernel_size=3, stride=2, padding=1),
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
        # self.conv0 = ConvInsBlock(in_channel, c, 3, 1)


    def forward(self, x):
        out0 = self.conv0_1(x) #64
        # print(out0.shape)
        out02 = self.conv0_2(out0)
        # print(out02.shape)
        out03 = self.conv0_3(out02)
        # print(out03.shape)
        out1 = self.conv1(out03)   # (Batch, 2c, H/2, W/2)  #128
        # print(out1.shape)
        out2 = self.conv2(out1)   # (Batch, 4c, H/4, W/4) #256
        # print(out2.shape)
        out3 = self.conv3(out2)   # (Batch, 8c, H/8, W/8) #512
        # print(out3.shape)
        return [out0, out1, out2, out3]


class Decoder(nn.Module):
    """
    Decoder for 2D images with skip connections.
    """

    def __init__(self, out_channel=1, first_out_channel=64, dropout_prob=0.3):
        super(Decoder, self).__init__()

        c = first_out_channel

        # 上采样块，从 8c -> 4c
        self.upconv3 = UpConvBlock(8 * c, 4 * c, kernel_size=4, stride=2, alpha=0.1)
        # 结合跳跃连接后的卷积块
        self.conv3 = ConvInsBlock(8 * c, 4 * c, kernel_size=3, stride=1, padding=1)
        # 残差块
        self.res3 = ResBlock(4 * c)
        # Dropout layer
        self.dropout3 = nn.Dropout2d(dropout_prob)

        # 上采样块，从 4c -> 2c
        self.upconv2 = UpConvBlock(4 * c, 2 * c, kernel_size=4, stride=2, alpha=0.1)
        # 结合跳跃连接后的卷积块
        self.conv2 = ConvInsBlock(4 * c, 2 * c, kernel_size=3, stride=1, padding=1)
        # 残差块
        self.res2 = ResBlock(2 * c)
        # Dropout layer
        self.dropout2 = nn.Dropout2d(dropout_prob)

        # 上采样块，从 2c -> c
        self.upconv1 = UpConvBlock(2 * c, c, kernel_size=4, stride=2, alpha=0.1)
        # 结合跳跃连接后的卷积块
        self.conv1 = ConvInsBlock(2 * c, c, kernel_size=3, stride=1, padding=1)
        # 残差块
        self.res1 = ResBlock(c)
        # Dropout layer
        self.dropout1 = nn.Dropout2d(dropout_prob)

        # 最终输出卷积层，将通道数调整为输出通道数
        self.conv0 = nn.Conv2d(c, out_channel, kernel_size=1, stride=1, padding=0)
        self.act = nn.Tanh()
        self.Film0 = FiLMModulation(input_dim=3, output_dim=8 * c)
        self.Film1 = FiLMModulation(input_dim=3, output_dim=4 * c)
        self.Film2 = FiLMModulation(input_dim=3, output_dim=2 * c)

    def forward(self, encoder_outputs, c_org):
        """
        encoder_outputs: list of encoder feature maps [out0, out1, out2, out3]
        """
        out0, out1, out2, out3 = encoder_outputs

        y1 = self.Film0(out3, c_org)
        # 第四层解码：上采样并结合跳跃连接
        up3 = self.upconv3(y1)  # (Batch, 4c, H/4, W/4)
        up3 = self.Film1(up3, c_org)
        # 拼接跳跃连接的特征图 out2
        concat3 = torch.cat([up3, out2], dim=1)  # (Batch, 8c, H/4, W/4)
        conv3 = self.conv3(concat3)  # (Batch, 4c, H/4, W/4)
        res3 = self.res3(conv3)  # (Batch, 4c, H/4, W/4)
        res3 = self.dropout3(res3)  # Apply Dropout

        # 第三层解码：上采样并结合跳跃连接
        up2 = self.upconv2(res3)  # (Batch, 2c, H/2, W/2)
        up2 = self.Film2(up2, c_org)
        # 拼接跳跃连接的特征图 out1
        concat2 = torch.cat([up2, out1], dim=1)  # (Batch, 4c, H/2, W/2)
        conv2 = self.conv2(concat2)  # (Batch, 2c, H/2, W/2)
        res2 = self.res2(conv2)  # (Batch, 2c, H/2, W/2)
        res2 = self.dropout2(res2)  # Apply Dropout

        # 第二层解码：上采样并结合跳跃连接
        up1 = self.upconv1(res2)  # (Batch, c, H, W)
        # 拼接跳跃连接的特征图 out0
        concat1 = torch.cat([up1, out0], dim=1)  # (Batch, 2c, H, W)
        conv1 = self.conv1(concat1)  # (Batch, c, H, W)
        res1 = self.res1(conv1)  # (Batch, c, H, W)
        res1 = self.dropout1(res1)  # Apply Dropout

        # 最终输出
        out = self.act(self.conv0(res1))  # (Batch, out_channel, H, W)

        return out


class RDP(nn.Module):
    def __init__(self, inshape=(192, 192), flow_multiplier=1., in_channel=1, channels=16):
        super(RDP, self).__init__()
        self.flow_multiplier = flow_multiplier
        self.channels = channels
        self.step = 7
        self.inshape = inshape

        c = self.channels
        # self.encoder_fixed = Encoder(in_channel=in_channel, first_out_channel=c)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_bilin = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        en_c=64
        self.warp = nn.ModuleList()
        self.diff = nn.ModuleList()
        self.conv_0 = ConvInsBlock(en_c, 16, 3, 1)
        self.conv_1 = ConvInsBlock(2 * en_c, 32, 3, 1)
        self.conv_2 = ConvInsBlock(4 * en_c, 64, 3, 1)
        self.conv_3 = ConvInsBlock(8 * en_c, 128, 3, 1)
        for i in range(4):
            self.warp.append(SpatialTransformer([s // 2 ** i for s in inshape]))
            self.diff.append(VecInt([s // 2 ** i for s in inshape]))

        # bottleNeck
        self.cconv_4 = nn.Sequential(
            ConvInsBlock(16 * c, 8 * c, 3, 1),
            ConvInsBlock(8 * c, 8 * c, 3, 1)
        )
        # warp scale 2
        self.defconv4 = nn.Conv2d(8 * c, 2, 3, 1, 1)
        self.defconv4.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv4.weight.shape))
        self.defconv4.bias = nn.Parameter(torch.zeros(self.defconv4.bias.shape))
        self.dconv4 = nn.Sequential(
            ConvInsBlock(3 * 8 * c, 8 * c),
            ConvInsBlock(8 * c, 8 * c)
        )

        self.upconv3 = UpConvBlock(8 * c, 4 * c, 4, 2)
        self.cconv_3 = CConv(3 * 4 * c)

        # warp scale 1
        self.defconv3 = nn.Conv2d(3 * 4 * c, 2, 3, 1, 1)
        self.defconv3.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv3.weight.shape))
        self.defconv3.bias = nn.Parameter(torch.zeros(self.defconv3.bias.shape))
        self.dconv3 = ConvInsBlock(3 * 4 * c, 4 * c)

        self.upconv2 = UpConvBlock(3 * 4 * c, 2 * c, 4, 2)
        self.cconv_2 = CConv(3 * 2 * c)

        # warp scale 0
        self.defconv2 = nn.Conv2d(3 * 2 * c, 2, 3, 1, 1)
        self.defconv2.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv2.weight.shape))
        self.defconv2.bias = nn.Parameter(torch.zeros(self.defconv2.bias.shape))
        self.dconv2 = ConvInsBlock(3 * 2 * c, 2 * c)

        self.upconv1 = UpConvBlock(3 * 2 * c, c, 4, 2)
        self.cconv_1 = CConv(3 * c)

        # decoder layers
        self.defconv1 = nn.Conv2d(3 * c, 2, 3, 1, 1)
        self.defconv1.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv1.weight.shape))
        self.defconv1.bias = nn.Parameter(torch.zeros(self.defconv1.bias.shape))

    def forward(self, moving_outputs, fixed_outputs,moving):
        # encode stage
        # M1, M2, M3, M4 = self.encoder_moving(moving)
        M1, M2, M3, M4 = moving_outputs
        F1, F2, F3, F4 = fixed_outputs
        # print(M1.shape,M2.shape,M3.shape,M4.shape)
        M1=self.conv_0(M1)
        M2 = self.conv_1(M2)
        M3 = self.conv_2(M3)
        M4 = self.conv_3(M4)

        F1=self.conv_0(F1)
        F2 = self.conv_1(F2)
        F3 = self.conv_2(F3)
        F4 = self.conv_3(F4)

        # first dec layer
        C4 = torch.cat([F4, M4], dim=1)
        C4 = self.cconv_4(C4)
        flow = self.defconv4(C4)
        flow = self.diff[3](flow)
        warped = self.warp[3](M4, flow)
        warped_4=warped
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
        flow = self.warp[1](flow, w) + w
        warped = self.warp[1](M2, flow)
        warped_2 = warped
        D2 = self.dconv2(C2)
        C2 = self.cconv_2(F2, warped, D2)
        v = self.defconv2(C2)
        w = self.diff[1](v)


        D1 = self.upconv1(C2)
        flow = self.upsample_bilin(2 * (self.warp[1](flow, w) + w))
        warped = self.warp[0](M1, flow)
        C1 = self.cconv_1(F1, warped, D1)
        v = self.defconv1(C1)
        w = self.diff[0](v)
        flow = self.warp[0](flow, w) + w
        warped_1 = warped
        y_moved = self.warp[0](moving, flow)
        # print(flow.shape)
        # print(warped.shape)

        return y_moved, flow

if __name__ == '__main__':
    # Create a square and a triangle image
    size = (1, 1, 192, 192)
    square = torch.zeros(size)
    triangle = torch.zeros(size)

    enc=Encoder()
    fea=enc(square)
    c_org=torch.randn((1, 3))
    dec=Decoder()
    y=dec(fea,c_org)
    print(y.shape)
    # print(enc)
    # print(dec)
    print("Content E", sum(p.numel() for p in enc.parameters() if p.requires_grad))
    print("Generator", sum(p.numel() for p in dec.parameters() if p.requires_grad))

    #
    # model = RDP(inshape=size[2:])
    # a,b,c,d=model(fea,fea,square,square)



