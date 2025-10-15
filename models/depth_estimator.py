import torch
import torch.nn as nn

class LightweightDepthEstimator(nn.Module):
    """
    Efficient pseudo-depth estimation from RGB
    Based on knowledge distillation from MiDaS
    """
    
    def __init__(self, input_channels=3, output_channels=1):
        super().__init__()
        
        # Encoder (MobileNetV3-inspired)
        self.encoder = nn.ModuleList([
            # Stage 1: 224x224 -> 112x112
            ConvBNReLU(input_channels, 16, kernel=3, stride=2),
            InvertedResidual(16, 24, kernel=3, stride=1, expand_ratio=1),
            
            # Stage 2: 112x112 -> 56x56
            InvertedResidual(24, 32, kernel=3, stride=2, expand_ratio=4),
            InvertedResidual(32, 32, kernel=3, stride=1, expand_ratio=3),
            
            # Stage 3: 56x56 -> 28x28
            InvertedResidual(32, 48, kernel=5, stride=2, expand_ratio=3),
            InvertedResidual(48, 48, kernel=5, stride=1, expand_ratio=3),
            
            # Stage 4: 28x28 -> 14x14
            InvertedResidual(48, 96, kernel=5, stride=2, expand_ratio=6),
            InvertedResidual(96, 96, kernel=5, stride=1, expand_ratio=6),
        ])
        
        # Decoder with skip connections
        self.decoder = nn.ModuleList([
            # 14x14 -> 28x28
            UpConvBlock(96, 48, scale=2),
            # 28x28 -> 56x56
            UpConvBlock(48 + 48, 32, scale=2),  # +48 from skip
            # 56x56 -> 112x112
            UpConvBlock(32 + 32, 16, scale=2),  # +32 from skip
            # 112x112 -> 224x224
            UpConvBlock(16 + 24, 8, scale=2),   # +24 from skip
        ])
        
        # Final depth prediction
        self.depth_head = nn.Sequential(
            nn.Conv2d(8, output_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Normalize depth to [0, 1]
        )
        
        # Multi-scale depth consistency
        self.multiscale_weights = nn.Parameter(torch.ones(4) * 0.25)
    
    def forward(self, x):
        # Encoder forward with skip connections
        skip_connections = []
        
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in [1, 3, 5]:  # Save features for skip connections
                skip_connections.append(x)
        
        # Decoder with skip connections
        for i, layer in enumerate(self.decoder):
            if i > 0 and i <= len(skip_connections):
                # Concatenate skip connection
                skip = skip_connections[-(i)]
                x = torch.cat([x, skip], dim=1)
            x = layer(x)
        
        # Predict depth
        depth = self.depth_head(x)
        
        return depth

class InvertedResidual(nn.Module):
    """Efficient inverted residual block for depth estimation"""
    def __init__(self, in_channels, out_channels, kernel, stride, expand_ratio):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
        
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channels, hidden_dim, kernel=1))
        
        layers.extend([
            # Depthwise conv
            nn.Conv2d(hidden_dim, hidden_dim, kernel, stride, kernel//2, groups=hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Pointwise conv
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)