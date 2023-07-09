from torchvision.models import vgg16_bn
from torch_snippets import *
import torch.nn as nn
from torchvision import transforms

class UNet(nn.Module):

    def __init__(self, pretrained=True, out_channels=12):
        super().__init__()

        ###################################
        ###Please enter your codes here####
        ###################################
        backbone = vgg16_bn(pretrained=pretrained).features
        self.layer0 = backbone[:6]  # 64, 224, 224
        self.layer1 = backbone[6:13]  # 128, 112, 112
        self.layer2 = backbone[13:23]  # 256 56, 56
        self.layer3 = backbone[23:33]  # 512 28 28
        self.layer4 = backbone[33:43]  # 512 14 14
        self.layer5 = backbone[43:44]  # 512 7 7
        self.up_conv4 = nn.ConvTranspose2d(
            512, 512, kernel_size=2, stride=2
        )
        self.up_conv3 = nn.ConvTranspose2d(
            512 * 2, 512, kernel_size=2, stride=2
        )
        self.up_conv2 = nn.ConvTranspose2d(
            512 * 2, 256, kernel_size=2, stride=2
        )
        self.up_conv1 = nn.ConvTranspose2d(
            256 * 2, 128, kernel_size=2, stride=2
        )

        self.up_conv0 = nn.ConvTranspose2d(
            128 * 2, 64, kernel_size=2, stride=2
        )
        self.conv = nn.Conv2d(128, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        ####################################

    def forward(self, x):
        ###################################
        ###Please enter your codes here####
        ###################################
        out_0 = self.layer0(x)
        out_1 = self.layer1(out_0)
        out_2 = self.layer2(out_1)
        out_3 = self.layer3(out_2)
        out_4 = self.layer4(out_3)
        out_5 = self.layer5(out_4)

        up4 = self.up_conv4(out_5)  # 512 14 14
        up4 = torch.cat([up4, out_4], dim=1)  # 1028 14 14

        up3 = self.up_conv3(up4)  # 512 28 28
        up3 = torch.cat([up3, out_3], dim=1)  # 1028 28 28

        up2 = self.up_conv2(up3)  # 256 56 56
        up2 = torch.cat([up2, out_2], dim=1)  # 512 56 56

        up1 = self.up_conv1(up2)  # 128 112 112
        up1 = torch.cat([up1, out_1], dim=1)  # 256 112 112

        x = self.up_conv0(up1)  # 64 224 224
        x = torch.cat([x, out_0], dim=1)  # 128 224 224
        x = self.conv(x)

        ####################################

        return x