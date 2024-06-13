import torch
from torch import nn
from .networks import resnet50


def give_encoder(model):
    # takingin resnet50 give_encoder
    model = resnet50(num_classes=1000)
    checkpoint = torch.load("resnet50_best.tar", map_location="cuda")
    pretrained_dict = checkpoint["state_dict"]
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k,
                       v in pretrained_dict.items() if k in model_dict}

    model.load_state_dict(pretrained_dict)

    return nn.Sequential(*list(model.children())[:-4])


class CVSaliency(nn.Module):
    def __init__(self, in_channels=1024*2, bilinear=True):
        super().__init__()
        self.encoder = give_encoder(resnet50())
        factor = 2 if bilinear else 1
        self.up1 = (Up(in_channels, 512 // factor, bilinear))
        self.up2 = (Up(256, 256 // factor, bilinear))
        self.up3 = (Up(128, 128 // factor, bilinear))
        self.up4 = (Up(64, 64, bilinear))
        self.outc = (OutConv(64, 1))

    def forward(self, x):

        x_feat = self.encoder(x)
        # print(f"Shape of x features: {x_feat.shape}")
        x_feat = torch.cat([x_feat.real, x_feat.imag], dim=1)
        x_up = self.up1(x_feat)
        x_up = self.up2(x_up)
        x_up = self.up3(x_up)
        x_up = self.up4(x_up)
        logits = self.outc(x_up)
        # print(f"Shape of output: {logits.shape}")

        return logits


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1):
        x1 = self.up(x1)
        # input is CHW
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]
        #
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        # x = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
