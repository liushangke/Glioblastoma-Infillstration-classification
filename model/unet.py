import torch.nn as nn
import torch

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

class UNet3DPatchBased(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Downsample
        self.dconv_down1 = double_conv(in_channels, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        
        self.maxpool = nn.MaxPool3d(2)
        
        # Upsample
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        # Final output
        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(out_channels, 3)

    def forward(self, x):
        # Downsample
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        
        # Upsample
        x = self.upsample(conv3)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)
        
        # Final output
        x = self.final_conv(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # flatten the tensor
        out = self.fc(x)

        return out
