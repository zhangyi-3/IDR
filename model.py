import torch
import torch.nn as nn


class UNet_n2n_un(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet_n2n_un, self).__init__()

        self.en_block1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(48, 48, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.MaxPool2d(2))

        self.en_block2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.MaxPool2d(2))

        self.en_block3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.MaxPool2d(2))

        self.en_block4 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.MaxPool2d(2))

        self.en_block5 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 48, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.de_block1 = nn.Sequential(
            nn.Conv2d(96, 96, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(96, 96, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.de_block2 = nn.Sequential(
            nn.Conv2d(144, 96, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(96, 96, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.de_block3 = nn.Sequential(
            nn.Conv2d(144, 96, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(96, 96, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.de_block4 = nn.Sequential(
            nn.Conv2d(144, 96, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(96, 96, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.de_block5 = nn.Sequential(
            nn.Conv2d(96 + in_channels, 64, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(64, 32, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),

            nn.Conv2d(32, out_channels, 3, padding=1, bias=True))

        # Initialize weights
        # self._init_weights()

    # def _init_weights(self):
    #     """Initializes weights using He et al. (2015)."""
    #     for m in self.modules():
    #         if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight.data)
    #             m.bias.data.zero_()

    def forward(self, x):
        pool1 = self.en_block1(x)
        pool2 = self.en_block2(pool1)
        pool3 = self.en_block3(pool2)
        pool4 = self.en_block4(pool3)
        upsample5 = self.en_block5(pool4)

        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self.de_block1(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self.de_block2(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self.de_block3(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self.de_block4(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)
        out = self.de_block5(concat1)
        return out
