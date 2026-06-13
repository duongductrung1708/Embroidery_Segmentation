import torch
import torch.nn as nn

# ==========================================
# CỤC LEGO CƠ BẢN: 2 lần Tích chập liên tiếp
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            # Mũi tên xanh dương thứ 1
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Mũi tên xanh dương thứ 2
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# ==========================================
# MÔ HÌNH U-NET HOÀN CHỈNH
# ==========================================
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        # in_channels = 3 (Ảnh màu RGB)
        # out_channels = 3 (3 Nhãn: Nền, Satin, Tatami)
        super(UNet, self).__init__()
        
        # MÁY NÉN (ENCODER - Đi xuống)
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # Mũi tên màu đỏ (Nén)
        
        # ĐÁY CHỮ U (BOTTLENECK)
        self.bottleneck = DoubleConv(512, 1024)
        
        # MÁY BƠM (DECODER - Đi lên)
        # Mũi tên xanh lá cây (Phóng to)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(1024, 512) # 1024 vì = 512 (dưới lên) + 512 (Skip connection)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv4 = DoubleConv(128, 64)
        
        # LỚP CUỐI CÙNG (OUTPUT MASK)
        # Mũi tên màu Cyan (Tích chập 1x1)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # --- NỬA TRÁI (ENCODER) ---
        x1 = self.down1(x)    # Lưu lại x1 để làm Skip Connection
        p1 = self.pool(x1)
        
        x2 = self.down2(p1)   # Lưu lại x2
        p2 = self.pool(x2)
        
        x3 = self.down3(p2)   # Lưu lại x3
        p3 = self.pool(x3)
        
        x4 = self.down4(p3)   # Lưu lại x4
        p4 = self.pool(x4)
        
        # --- ĐÁY ---
        b = self.bottleneck(p4)
        
        # --- NỬA PHẢI (DECODER) ---
        up_1 = self.up1(b)
        concat_1 = torch.cat((x4, up_1), dim=1) # Mũi tên xám: Ghép/Nối kênh
        d1 = self.up_conv1(concat_1)
        
        up_2 = self.up2(d1)
        concat_2 = torch.cat((x3, up_2), dim=1) # Mũi tên xám
        d2 = self.up_conv2(concat_2)
        
        up_3 = self.up3(d2)
        concat_3 = torch.cat((x2, up_3), dim=1) # Mũi tên xám
        d3 = self.up_conv3(concat_3)
        
        up_4 = self.up4(d3)
        concat_4 = torch.cat((x1, up_4), dim=1) # Mũi tên xám
        d4 = self.up_conv4(concat_4)
        
        # --- OUTPUT ---
        out = self.final_conv(d4)
        return out
