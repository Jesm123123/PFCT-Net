import torch.nn as nn
import torch.nn.functional as F
import torch
from .modules import  ASPP, RCB,  CBAM, GIFB, RFB
from .pvt_v2 import PyramidVisionTransformerV2
from functools import partial
from timm.models.vision_transformer import _cfg

pth = './pvt_v2_b3.pth'


# pth = r"D:\personal_project\CORE_Project\Models\pvt_v2_b3.pth"


class DoubleConv(nn.Module):
    """
    1. DoubleConv 模块
    (convolution => [BN] => ReLU) * 2
    连续两次的卷积操作：U-net网络中，下采样和上采样过程，每一层都会连续进行两次卷积操作
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # torch.nn.Sequential是一个时序容器，Modules 会以它们传入的顺序被添加到容器中。
        # 此处：卷积->BN->ReLU->卷积->BN->ReLU
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    2. Down(下采样)模块
    Downscaling with maxpool then double conv
    maxpool池化层，进行下采样，再接DoubleConv模块
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),  # 池化层
            DoubleConv(in_channels, out_channels)  # DoubleConv模块
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class PVT(nn.Module):
    def __init__(self):

        super().__init__()

        backbone = PyramidVisionTransformerV2(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 18, 3],
            sr_ratios=[8, 4, 2, 1],
        )

        checkpoint = torch.load(pth)
        backbone.default_cfg = _cfg()
        backbone.load_state_dict(checkpoint)
        self.backbone = torch.nn.Sequential(*list(backbone.children()))[:-1]

        for i in [1, 4, 7, 10]:
            self.backbone[i] = torch.nn.Sequential(*list(self.backbone[i].children()))

        self.LE = nn.ModuleList([])
        for i in range(4):
            self.LE.append(
                nn.Sequential(
                    RCB([64, 128, 320, 512][i], 64), RCB(64, 64)
                )
            )

        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_4 = nn.Upsample(scale_factor=4, mode='bicubic')

        self.TB_out1 = nn.Sequential(RCB(512 + 64, 512))
        self.TB_out2 = nn.Sequential(RCB(256 + 64, 256))
        self.TB_out3 = nn.Sequential(RCB(128 + 64, 128))
        self.TB_out4 = nn.Sequential(RCB(64 + 64, 64))

    def get_pyramid(self, x):
        pyramid = []
        B = x.shape[0]  # Batch_size
        for i, module in enumerate(self.backbone):
            if i in [0, 3, 6, 9]:
                x, H, W = module(x)
            elif i in [1, 4, 7, 10]:
                for sub_module in module:
                    x = sub_module(x, H, W)
            else:
                x = module(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                pyramid.append(x)

        return pyramid

    def forward(self, x, f1, f2, f3, f4):
        pyramid = self.get_pyramid(x)
        pyramid_emph = []
        for i, level in enumerate(pyramid):
            pyramid_emph.append(self.LE[i](pyramid[i]))

        pyramid_emph[-1] = self.up_2(self.up_2(pyramid_emph[-1]))
        pyramid_emph[-2] = self.up_2(self.up_2(pyramid_emph[-2]))
        pyramid_emph[-3] = self.up_4(pyramid_emph[-3])
        pyramid_emph[-4] = self.up_4(pyramid_emph[-4])

        l1 = torch.cat((pyramid_emph[-1], f1), dim=1)
        l2 = torch.cat((pyramid_emph[-2], f2), dim=1)
        l3 = torch.cat((pyramid_emph[-3], f3), dim=1)
        l4 = torch.cat((pyramid_emph[-4], f4), dim=1)
        return self.TB_out1(l1), self.TB_out2(l2), self.TB_out3(l3), self.TB_out4(l4)


class Up1(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # self.conv = DoubleConv(in_channels, out_channels)
        self.conv = RCB(in_channels, out_channels)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, out_channels, kernel_size=2, stride=2)

    def forward(self, x1):
        x1 = self.conv(x1)
        x1 = self.up(x1)
        return x1


class Up2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # self.conv = DoubleConv(in_channels, out_channels)
        self.conv = RCB(in_channels, out_channels)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        _, _, x1_h, x1_w = x1.size()
        x2 = F.interpolate(x2, size=(x1_h, x1_w), mode='bilinear', align_corners=False)
        concat = torch.cat([x1, x2], dim=1)
        concat = self.conv(concat)
        out = self.up(concat)
        return out


class PFCT(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(PFCT, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        self.CA1 = nn.Sequential(
            CBAM(64),
        )
        self.CA2 = nn.Sequential(
            CBAM(128),
        )
        self.CA3 = nn.Sequential(
            CBAM(256),
        )

        self.ASPP = ASPP(512, 512)

        # self.aggregation = aggregation(128)
        self.PVT = PVT()
        self.GIFB1 = GIFB(512, 256)
        self.GIFB2 = GIFB(256, 128)
        self.GIFB3 = GIFB(128, 64)

        self.RFB1 = RFB(512, 256)
        self.RFB2 = RFB(256, 128)
        self.RFB3 = RFB(128, 64)

        self.final = RCB(64, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        ASPP = self.ASPP(x4)
        # print(x4.shape)
        CA_x1 = self.CA1(x1)
        CA_x2 = self.CA2(x2)
        CA_x3 = self.CA3(x3)

        out1, out2, out3, out4 = self.PVT(x, ASPP, CA_x3, CA_x2, CA_x1)

        GIFB1 = self.GIFB1(out1, out2)
        GIFB2 = self.GIFB2(GIFB1, out3)
        GIFB3 = self.GIFB3(GIFB2, out4)

        RFB1 = self.RFB1(ASPP, x3)
        RFB2 = self.RFB2(RFB1, x2)
        RFB3 = self.RFB3(RFB2, x1)

        out = self.final(GIFB3 + RFB3)
        return out
