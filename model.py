from torchvision.models import vgg16_bn, resnet34, resnet50, resnet101
from torch_snippets import *
import torch.nn as nn
from torchvision import transforms

class UNet(nn.Module):

    def __init__(self, pretrained=True, out_channels=12, backbone = 'vgg16', with_boundary = False):
        super().__init__()
        self.with_boundary = with_boundary
        self.backbone = backbone
        if self.backbone == 'vgg16':
            self.vgg16_init(pretrained, out_channels)
        elif 'resnet' in self.backbone:
            self.resnet34_init(pretrained,out_channels, self.backbone)
            if with_boundary:
                self.boundary_decoder_init()
                self.fusion_decoder_init()
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

    def resnet34_init(self,pretrained,out_channels, backbone):
        kk = None
        if backbone == 'resnet34':
            self.extractor = resnet34(pretrained=pretrained)
            kk = 512
        elif backbone == 'resnet50':
            self.extractor = resnet50(pretrained=pretrained)
            kk = 2048
        elif backbone == 'resnet101':
            self.extractor = resnet101(pretrained=pretrained)
            kk = 2048
        



        self.up_conv4 = nn.ConvTranspose2d(
            kk, int(kk/2), kernel_size=2, stride=2
        )
        self.back_conv3 = nn.Sequential(
            nn.Conv2d(kk, int(kk/2), kernel_size=3, stride= 1, padding=1),
            nn.BatchNorm2d(int(kk/2)),
            nn.ReLU(inplace=True)
        )

        self.up_conv3 = nn.ConvTranspose2d(
            int(kk/2), int(kk/4), kernel_size=2, stride=2
        )
        self.back_conv2 = nn.Sequential(
            nn.Conv2d(int(kk/2), int(kk/4), kernel_size=3, stride= 1, padding=1),
            nn.BatchNorm2d(int(kk/4)),
            nn.ReLU(inplace=True)
        )

        self.up_conv2 = nn.ConvTranspose2d(
            int(kk/4), int(kk/8), kernel_size=2, stride=2
        )

        self.back_conv1 = nn.Sequential(
            nn.Conv2d(int(kk/4), int(kk/8), kernel_size=3, stride= 1, padding=1),
            nn.BatchNorm2d(int(kk/8)),
            nn.ReLU(inplace=True)
        )
        if backbone == 'resnet101' or backbone == 'resnet50':
            self.up_conv1 = nn.ConvTranspose2d(
                int(kk/8), int(kk/32), kernel_size=2, stride=2
            )

            self.back_conv0 = nn.Sequential(
                nn.Conv2d(int(kk/16), int(kk/32), kernel_size=3, stride= 1, padding=1),
                nn.BatchNorm2d(int(kk/32)),
                nn.ReLU(inplace=True)
            )

            self.up_conv0 = nn.ConvTranspose2d(
                int(kk/32), int(kk/64), kernel_size=2, stride=2
            )
            self.conv = nn.Conv2d(int(kk/64), out_channels, kernel_size=3, stride=1, padding=1)
        elif backbone == 'resnet34':
            self.up_conv1 = nn.ConvTranspose2d(
                int(kk/8), int(kk/8), kernel_size=2, stride=2
            )

            self.back_conv0 = nn.Sequential(
                nn.Conv2d(int(kk/4), int(kk/8), kernel_size=3, stride= 1, padding=1),
                nn.BatchNorm2d(int(kk/8)),
                nn.ReLU(inplace=True)
            )

            self.up_conv0 = nn.ConvTranspose2d(
                int(kk/8), int(kk/16), kernel_size=2, stride=2
            )
            self.conv = nn.Conv2d(int(kk/16), out_channels, kernel_size=3, stride=1, padding=1)
        # elif backbone == 'resnet50':
        #     self.up_conv1 = nn.ConvTranspose2d(
        #         int(kk/8), int(kk/8), kernel_size=2, stride=2
        #     )

        #     self.back_conv0 = nn.Sequential(
        #         nn.Conv2d(int(kk/4), int(kk/8), kernel_size=3, stride= 1, padding=1),
        #         nn.BatchNorm2d(int(kk/8)),
        #         nn.ReLU(inplace=True)
        #     )

        #     self.up_conv0 = nn.ConvTranspose2d(
        #         int(kk/8), int(kk/16), kernel_size=2, stride=2
        #     )
        #     self.conv = nn.Conv2d(int(kk/16), out_channels, kernel_size=3, stride=1, padding=1)


    def boundary_decoder_init(self):
        self.b_up_conv4 = nn.ConvTranspose2d(
            512, 256, kernel_size=2, stride=2
        )
        self.b_back_conv3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride= 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.b_up_conv3 = nn.ConvTranspose2d(
            256, 128, kernel_size=2, stride=2
        )
        self.b_back_conv2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride= 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.b_up_conv2 = nn.ConvTranspose2d(
            128, 64, kernel_size=2, stride=2
        )

        self.b_back_conv1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride= 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.b_up_conv1 = nn.ConvTranspose2d(
            64, 64, kernel_size=2, stride=2
        )

        self.b_back_conv0 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride= 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.b_up_conv0 = nn.ConvTranspose2d(
            64, 32, kernel_size=2, stride=2
        )
        self.b_conv = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)


    def fusion_decoder_init(self):
        self.f_up3 = nn.ConvTranspose2d(
            512, 256, kernel_size=2, stride=2
        )

        self.f_conv3_b = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride= 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.f_conv3_r = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride= 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.f_up2 = nn.ConvTranspose2d(
            256, 128, kernel_size=2, stride=2
        )

        self.f_conv2_b = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride= 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.f_conv2_r = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride= 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.f_up1 = nn.ConvTranspose2d(
            128, 64, kernel_size=2, stride=2
        )

        self.f_conv1_b = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride= 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.f_conv1_r = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride= 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.f_up0 = nn.ConvTranspose2d(
            64, 64, kernel_size=2, stride=2
        )
        self.f_conv0_b = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride= 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.f_conv0_r = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride= 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.f_up = nn.ConvTranspose2d(
            64, 32, kernel_size=2, stride=2
        )


    def resnet34_forward(self,x):
        out_0 = self.extractor.conv1(x)  # 112 112 64

        out_0 = self.extractor.bn1(out_0)
        out_0 = self.extractor.relu(out_0)

        out_1 = self.extractor.maxpool(out_0)  # 56 56 64
        out_1 = self.extractor.layer1(out_1)  # 56 56 64

        out_2 = self.extractor.layer2(out_1)  # 28 28 128

        out_3 = self.extractor.layer3(out_2)  # 14 14 256

        out_4 = self.extractor.layer4(out_3)  # 7 7 512
        # print("out0: ",out_0.shape)
        # print("out1: ",out_1.shape)
        # print("out2: ",out_2.shape)
        # print("out3: ",out_3.shape)
        # print("out4: ",out_4.shape)
        up3 = self.up_conv4(out_4)  # 14 14 256

        up3 = torch.cat([up3, out_3], dim=1)  # 14 14 512
        up3 = self.back_conv3(up3) # 14 14 256
        # print("up3: ",up3.shape)

        up2 = self.up_conv3(up3) #  28 28 128
        up2 = torch.cat([up2, out_2], dim=1) # 28 28 256
        up2 = self.back_conv2(up2) # 28 28 128
        # print("up2: ",up2.shape)
        up1 = self.up_conv2(up2) # 56 56 64
        up1 = torch.cat([up1, out_1], dim=1) # 56 56 128
        up1 = self.back_conv1(up1) # 56 56 64
        # print("up1: ",up1.shape)

        up0 = self.up_conv1(up1) # 112 112 64
        # print("-----------")
        # print(up0.shape, out_0.shape)
        up0 = torch.cat([up0, out_0], dim=1) # 112 112 128
        up0 = self.back_conv0(up0) # 112 112 64
        # print("up0: ",up0.shape)

        up = self.up_conv0(up0) # 224 224 32
        x = self.conv(up)
        return x
    
    def resnet34_boundary_forward(self, x):
        out_0 = self.extractor.conv1(x)  # 112 112 64

        out_0 = self.extractor.bn1(out_0)
        out_0 = self.extractor.relu(out_0)

        out_1 = self.extractor.maxpool(out_0)  # 56 56 64
        out_1 = self.extractor.layer1(out_1)  # 56 56 64

        out_2 = self.extractor.layer2(out_1)  # 28 28 128

        out_3 = self.extractor.layer3(out_2)  # 14 14 256

        out_4 = self.extractor.layer4(out_3)  # 7 7 512

        # boundary detector
        up3 = self.b_up_conv4(out_4)  # 14 14 256
        up3 = torch.cat([up3, out_3], dim=1)  # 14 14 512
        up3 = self.b_back_conv3(up3) # 14 14 256

        up2 = self.b_up_conv3(up3) #  28 28 128
        up2 = torch.cat([up2, out_2], dim=1) # 28 28 256
        up2 = self.b_back_conv2(up2) # 28 28 128

        up1 = self.b_up_conv2(up2) # 56 56 64
        up1 = torch.cat([up1, out_1], dim=1) # 56 56 128
        up1 = self.b_back_conv1(up1) # 56 56 64

        up0 = self.b_up_conv1(up1) # 112 112 64
        up0 = torch.cat([up0, out_0], dim=1) # 112 112 128
        up0 = self.b_back_conv0(up0) # 112 112 64

        up = self.b_up_conv0(up0) # 224 224 32
        b_x = self.b_conv(up)

        # room type decoder 
        r_up3 = self.up_conv4(out_4)  # 14 14 256
        r_up3 = torch.cat([r_up3, out_3], dim=1)  # 14 14 512
        r_up3 = self.back_conv3(r_up3) # 14 14 256

        # fusion channel
        f_out3 = self.f_up3(out_4) # 14 14 256 
        b_out3 = self.f_conv3_b(torch.cat([up3, r_up3], dim=1)) # 14 14 256 
        r_out3 = self.f_conv3_r(torch.cat([r_up3,b_out3], dim=1)) # 14 14 256 
        f_out3 = f_out3 + r_out3 + up3 + r_up3 # 14 14 256 

        # room type decoder 
        r_up2 =  self.up_conv3(f_out3)  # 28 28 128 
        r_up2 = torch.cat([r_up2, out_2], dim=1) # 28 28 256 
        r_up2 = self.back_conv2(r_up2) # 28 28 128 

        # fusion channel
        f_out2 = self.f_up2(f_out3) # 28 28 128 
        b_out2 = self.f_conv2_b(torch.cat([up2, r_up2], dim=1)) # 28 28 128 
        r_out2 = self.f_conv2_r(torch.cat([r_up2, b_out2],dim=1)) # 28 28 128 
        f_out2 = f_out2 + r_out2 + up2 + r_up2 # 28 28 128 

        # room type decoder 
        r_up1 =  self.up_conv2(f_out2)  #56 56 64
        r_up1 = torch.cat([r_up1, out_1], dim=1) # 56 56 128
        r_up1 = self.back_conv1(r_up1) # 56 56 64

        # fusion channel
        f_out1 = self.f_up1(f_out2)  #56 56 64
        b_out1 = self.f_conv1_b(torch.cat([up1, r_up1],dim=1)) #56 56 64
        r_out1 = self.f_conv1_r(torch.cat([r_up1, b_out1], dim=1)) #56 56 64
        f_out1 = f_out1 + r_out1 + up1 + r_up1 #56 56 64

        # room type decoder 
        r_up0 =  self.up_conv1(f_out1)  #112 112 64
        r_up0 = torch.cat([r_up0, out_0], dim=1) # 112 112 128
        r_up0 = self.back_conv0(r_up0) # 112 112 64

        # fusion channel
        f_out0 = self.f_up0(f_out1)  #112 112 64
        b_out0 = self.f_conv0_b(torch.cat([up0, r_up0],dim=1)) #112 112 64
        r_out0 = self.f_conv0_r(torch.cat([r_up0, b_out0], dim=1)) #112 112 64
        f_out0 = f_out0 + r_out0 + up0 + r_up0 #112 112 64

        f_out = self.f_up(f_out0)
        mask = self.conv(f_out)
        return mask, b_x


    def forward(self, x):
        if self.backbone == 'vgg16':
            return self.vgg16_forward(x)
        else:
            if self.with_boundary:
                return self.resnet34_boundary_forward(x)
            else:
                return self.resnet34_forward(x)
