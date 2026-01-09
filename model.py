import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from timm.models.swin_transformer import SwinTransformerBlock

# ----------------------------------------------
#  Enhanced Pyramid Pooling Module
# ----------------------------------------------
class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels=128, out_channels=32, bin_sizes=[1, 2, 3, 6]):
        super().__init__()
        self.features = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(bin_size),
                nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(in_channels//4, in_channels//4, 1)
            ) for bin_size in bin_sizes
        ])
        self.residual_proj = nn.Conv2d(in_channels, out_channels, 1)  # Add projection
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels + len(bin_sizes)*(in_channels//4), out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        orig = x
        pyramids = [x]
        for f in self.features:
            pyramid = f(x)
            pyramid = F.interpolate(pyramid, size=x.shape[2:], mode='bilinear')
            pyramids.append(pyramid)
        out = torch.cat(pyramids, dim=1)
        residual = self.residual_proj(orig)  # Project residual to match channels
        return self.fusion(out) + residual  # Now channels match
# ----------------------------------------------
#  Fixed Swin Transformer Fusion Module
# ----------------------------------------------
class SwinTransformerFusion(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=7, shift=False):
        super().__init__()
        self.window_size = window_size
        self.shift = shift
        self.num_heads = num_heads
        
        # Fix shift_size as tuple
        shift_size = (window_size//2, window_size//2) if shift else (0, 0)
        
        self.norm = nn.LayerNorm(2 * dim)
        self.reduce = nn.Conv2d(2 * dim, dim, kernel_size=3, padding=1)
        
        self.swin_block = SwinTransformerBlock(
            dim=2 * dim,
            input_resolution=(window_size, window_size),
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,  # Use tuple
            mlp_ratio=4.0,
            qkv_bias=True
        )

    def forward(self, x, g):
        B, C, H, W = x.shape
        combined = torch.cat([x, g], dim=1)
        combined = combined.permute(0, 2, 3, 1)
        
        # Window size adaptation
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h + pad_w > 0:
            combined = F.pad(combined, (0, 0, 0, pad_w, 0, pad_h))
        
        self.swin_block.input_resolution = (H + pad_h, W + pad_w)
        
        # Process through Swin
        normalized = self.norm(combined)
        attended = self.swin_block(normalized)
        
        # Remove padding and restore shape
        if pad_h + pad_w > 0:
            attended = attended[:, :H, :W, :]
        attended = attended.permute(0, 3, 1, 2)
        
        return self.reduce(attended) + x  # Residual connection

# ----------------------------------------------
#  Enhanced Feature Fusion
# ----------------------------------------------
class FeatureFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention1 = SwinTransformerFusion(channels, shift=True)
        self.attention2 = SwinTransformerFusion(channels, shift=False)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, x, g):
        x = x + self.attention1(x, g)  # Enhanced residual
        x = x + self.attention2(x, x)  # Dual attention
        return self.conv(x)

class SqueezeExcitation(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction),
            nn.GELU(),
            nn.Linear(channel//reduction, channel),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
# ----------------------------------------------
#  Stable UNet Decoder
# ----------------------------------------------

class UNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels[3], out_channels, 3, padding=1)
        )
        self.fusion1 = FeatureFusion(out_channels)
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        self.fusion2 = FeatureFusion(out_channels)
        
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        self.fusion3 = FeatureFusion(out_channels)

    def forward(self, features):
        f1, f2, f3, f4 = features
        x = self.fusion1(self.up1(f4), f3)
        x = self.fusion2(self.up2(x), f2)
        x = self.fusion3(self.up3(x), f1)
        return x
# ----------------------------------------------
#  Final Network Architecture
# ----------------------------------------------
class Network(nn.Module):
    def __init__(self, channel=32, imagenet_pretrained=True, mode=True):
        super(Network, self).__init__()
        self.backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1, mode='rgb')
        self.ppm = PyramidPoolingModule(in_channels=128,  out_channels=32)
        # Feature compression layers
        self.conv96to32 = nn.Sequential(
            nn.Conv2d(192, 48, 3, padding=1),  # For x1 (96 channels)
            nn.BatchNorm2d(48),
            nn.PReLU(), #gelu
            nn.Conv2d(48, 32, 1)
        )

        self.conv192to32 = nn.Sequential( 
            nn.Conv2d(384, 96, 3, padding=1),  # For x2_x3 (192+192=384 channels)
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 32, 1)
        )

        self.conv384to32 = nn.Sequential(
            nn.Conv2d(768, 192, 3, padding=1),  # For x4_x5 (384+384=768 channels)
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 32, 1)
        )


        self.conv768to32 = nn.Sequential(
            nn.Conv2d(1536, 384, 3, padding=1),  # For x6_x7 (768+768=1536 channels)
            nn.BatchNorm2d(384),
            nn.PReLU(),
            nn.Conv2d(384, 128, 3, padding=1),  # For x1 (96 channels)
            nn.PReLU()
        )

        
        # Batch norms for summation stability
        self.bn_f1 = nn.BatchNorm2d(32)
        self.bn_f2 = nn.BatchNorm2d(32)
        self.bn_f3 = nn.BatchNorm2d(32)
        self.bn_f4 = nn.BatchNorm2d(128)
        
        # Decoder with stable features
        self.decoder = UNetDecoder([channel]*4, channel)
        
        # Prediction head
        self.head = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GELU(),
            SqueezeExcitation(32),  # Added attention
            nn.Conv2d(32, 1, 1)
        )
    def forward(self, x):
        # Backbone features
        x = self.backbone.features[0](x)
        x1 = self.backbone.features[1](x)
        x2 = self.backbone.features[2](x1)
        x3 = self.backbone.features[3](x2)
        x4 = self.backbone.features[4](x3)
        x5 = self.backbone.features[5](x4)
        x6 = self.backbone.features[6](x5)
        x7 = self.backbone.features[7](x6)
        
        # Feature compression
        x_x1 = torch.cat([x, x1], dim=1)
        f1 = self.bn_f1(self.conv96to32(x_x1))
        x2_x3 = torch.cat([x2, x3], dim=1)  # Concatenate along channels
        f2 = self.bn_f2(self.conv192to32(x2_x3))
        # Apply similarly for other stages:
        x4_x5 = torch.cat([x4, x5], dim=1)
        f3 = self.bn_f3(self.conv384to32(x4_x5))

        x6_x7 = torch.cat([x6, x7], dim=1)
        f4 = self.ppm(self.bn_f4(self.conv768to32(x6_x7)))
        
        # Decoding
        x = self.decoder([f1, f2, f3, f4])
        return F.interpolate(self.head(x), scale_factor=4, mode='bilinear')
