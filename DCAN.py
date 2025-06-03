import torch
import torch.nn as nn

class DCAN(nn.Module):
    def __init__(self, args):
        super(DCAN, self).__init__()
        
        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        in_ch = 3 if args.rgb else 1
        # Encoder
        self.enc1 = CBR(in_ch, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = CBR(512, 1024)

        # Decoder for object mask
        self.upconv4_obj = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4_obj = CBR(1024, 512)
        self.upconv3_obj = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3_obj = CBR(512, 256)
        self.upconv2_obj = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2_obj = CBR(256, 128)
        self.upconv1_obj = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1_obj = CBR(128, 64)
        self.conv_out_obj = nn.Conv2d(64, 1, kernel_size=1)

        # Decoder for contour
        self.upconv4_contour = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4_contour = CBR(1024, 512)
        self.upconv3_contour = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3_contour = CBR(512, 256)
        self.upconv2_contour = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2_contour = CBR(256, 128)
        self.upconv1_contour = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1_contour = CBR(128, 64)
        self.conv_out_contour = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        # Decoder for object mask
        d4_obj = self.dec4_obj(torch.cat([self.upconv4_obj(b), e4], dim=1))
        d3_obj = self.dec3_obj(torch.cat([self.upconv3_obj(d4_obj), e3], dim=1))
        d2_obj = self.dec2_obj(torch.cat([self.upconv2_obj(d3_obj), e2], dim=1))
        d1_obj = self.dec1_obj(torch.cat([self.upconv1_obj(d2_obj), e1], dim=1))
        out_obj = torch.sigmoid(self.conv_out_obj(d1_obj))

        # Decoder for contour
        d4_contour = self.dec4_contour(torch.cat([self.upconv4_contour(b), e4], dim=1))
        d3_contour = self.dec3_contour(torch.cat([self.upconv3_contour(d4_contour), e3], dim=1))
        d2_contour = self.dec2_contour(torch.cat([self.upconv2_contour(d3_contour), e2], dim=1))
        d1_contour = self.dec1_contour(torch.cat([self.upconv1_contour(d2_contour), e1], dim=1))
        out_contour = torch.sigmoid(self.conv_out_contour(d1_contour))

        return out_obj, out_contour

# Example usage
class Args:
    rgb = False

model = DCAN(Args())
x = torch.randn((1, 1, 256, 256))
out_obj, out_contour = model(x)
print("Output shapes:", out_obj.shape, out_contour.shape)
