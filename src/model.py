import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2):
        super(UNet, self).__init__()
        
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.bottleneck = DoubleConv(512, 1024)
        
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(1024, 512) 
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv4 = DoubleConv(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)    
        p1 = self.pool(x1)
        x2 = self.down2(p1)   
        p2 = self.pool(x2)
        x3 = self.down3(p2)   
        p3 = self.pool(x3)
        x4 = self.down4(p3)   
        p4 = self.pool(x4)
        
        b = self.bottleneck(p4)
        
        up_1 = self.up1(b)
        concat_1 = torch.cat((x4, up_1), dim=1) 
        d1 = self.up_conv1(concat_1)
        
        up_2 = self.up2(d1)
        concat_2 = torch.cat((x3, up_2), dim=1) 
        d2 = self.up_conv2(concat_2)
        
        up_3 = self.up3(d2)
        concat_3 = torch.cat((x2, up_3), dim=1) 
        d3 = self.up_conv3(concat_3)
        
        up_4 = self.up4(d3)
        concat_4 = torch.cat((x1, up_4), dim=1) 
        d4 = self.up_conv4(concat_4)
        
        out = self.final_conv(d4)
        return out