from torchvision.models import vgg16_bn, resnet34
from torch_snippets import *
import torch.nn as nn
from torchvision import transforms

class UNet(nn.Module):

    def __init__(self, pretrained=True, out_channels=12, backbone = 'vgg16'):
        super().__init__()

        self.backbone = backbone
        if self.backbone == 'vgg16':
            self.vgg16_init(pretrained, out_channels)
        elif self.backbone == 'resnet34':
            self.resnet34_init(pretrained,out_channels)
        else:
            None

    def vgg16_init(self, pretrained, out_channels):
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

    def vgg16_forward(self, x):
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
        return x

    def resnet34_init(self,pretrained,out_channels):
        self.extractor = resnet34(pretrained=pretrained)

        self.up_conv4 = nn.ConvTranspose2d(
            512, 256, kernel_size=2, stride=2
        )
        self.back_conv3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride= 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.up_conv3 = nn.ConvTranspose2d(
            256, 128, kernel_size=2, stride=2
        )
        self.back_conv2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride= 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.up_conv2 = nn.ConvTranspose2d(
            128, 64, kernel_size=2, stride=2
        )

        self.back_conv1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride= 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.up_conv1 = nn.ConvTranspose2d(
            64, 64, kernel_size=2, stride=2
        )

        self.back_conv0 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride= 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.up_conv0 = nn.ConvTranspose2d(
            64, 32, kernel_size=2, stride=2
        )
        self.conv = nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1)

    def resnet34_forward(self,x):
        out_0 = self.extractor.conv1(x)  # 112 112 64

        out_0 = self.extractor.bn1(out_0)
        out_0 = self.extractor.relu(out_0)

        out_1 = self.extractor.maxpool(out_0)  # 56 56 64
        out_1 = self.extractor.layer1(out_1)  # 56 56 64

        out_2 = self.extractor.layer2(out_1)  # 28 28 128

        out_3 = self.extractor.layer3(out_2)  # 14 14 256

        out_4 = self.extractor.layer4(out_3)  # 7 7 512

        up3 = self.up_conv4(out_4)  # 14 14 256
        up3 = torch.cat([up3, out_3], dim=1)  # 14 14 512
        up3 = self.back_conv3(up3) # 14 14 256

        up2 = self.up_conv3(up3) #  28 28 128
        up2 = torch.cat([up2, out_2], dim=1) # 28 28 256
        up2 = self.back_conv2(up2) # 28 28 128

        up1 = self.up_conv2(up2) # 56 56 64
        up1 = torch.cat([up1, out_1], dim=1) # 56 56 128
        up1 = self.back_conv1(up1) # 56 56 64

        up0 = self.up_conv1(up1) # 112 112 64
        up0 = torch.cat([up0, out_0], dim=1) # 112 112 128
        up0 = self.back_conv0(up0) # 112 112 64

        up = self.up_conv0(up0) # 224 224 32
        x = self.conv(up)
        return x

    def forward(self, x):
        if self.backbone == 'vgg16':
            return self.vgg16_forward(x)
        else:
            return self.resnet34_forward(x)