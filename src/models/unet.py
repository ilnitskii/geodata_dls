import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(Conv2d => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """MaxPool => DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """ConvTranspose2d => DoubleConv (без выравнивания размеров)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Конкатенация без выравнивания (размеры должны совпадать)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Энкодер
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # Декодер
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        # Выходной слой
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Энкодер
        x1 = self.inc(x)        # [B, 3, 512, 512] -> [B, 64, 512, 512]
        x2 = self.down1(x1)     # [B, 64, 512, 512] -> [B, 128, 256, 256]
        x3 = self.down2(x2)     # [B, 128, 256, 256] -> [B, 256, 128, 128]
        x4 = self.down3(x3)     # [B, 256, 128, 128] -> [B, 512, 64, 64]
        x5 = self.down4(x4)     # [B, 512, 64, 64] -> [B, 1024, 32, 32]

        # Декодер
        x = self.up1(x5, x4)    # [B, 1024, 32, 32] -> [B, 512, 64, 64]
        x = self.up2(x, x3)     # [B, 512, 64, 64] -> [B, 256, 128, 128]
        x = self.up3(x, x2)     # [B, 256, 128, 128] -> [B, 128, 256, 256]
        x = self.up4(x, x1)     # [B, 128, 256, 256] -> [B, 64, 512, 512]

        # Выход
        logits = self.outc(x)   # [B, 64, 512, 512] -> [B, 1, 512, 512]
        return logits