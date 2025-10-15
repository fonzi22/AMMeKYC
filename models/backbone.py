import torch.nn as nn

class SharedFaceEncoder(nn.Module):
    """
    Lightweight face encoder optimized for mobile devices
    Shares weights between reference and video branches
    """
    
    def __init__(self, backbone='mobilefacenet', embedding_dim=256):
        super().__init__()
        
        if backbone == 'mobilefacenet':
            # MobileFaceNet architecture for lightweight extraction
            self.backbone = nn.Sequential(
                # Initial conv block
                ConvBlock(3, 64, kernel=3, stride=2, padding=1),  # 112x112 -> 56x56
                
                # Depthwise separable blocks
                DepthwiseSeparableBlock(64, 64, stride=1),
                
                # Bottleneck blocks
                BottleneckBlock(64, 128, stride=2, expansion=2),  # 56x56 -> 28x28
                BottleneckBlock(128, 128, stride=1, expansion=4),
                BottleneckBlock(128, 128, stride=1, expansion=2),
                
                BottleneckBlock(128, 256, stride=2, expansion=4),  # 28x28 -> 14x14
                BottleneckBlock(256, 256, stride=1, expansion=2),
                BottleneckBlock(256, 256, stride=1, expansion=4),
                BottleneckBlock(256, 256, stride=1, expansion=2),
                
                BottleneckBlock(256, 512, stride=2, expansion=4),  # 14x14 -> 7x7
                BottleneckBlock(512, 512, stride=1, expansion=2),
                
                # Global pooling
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            
        elif backbone == 'efficientnet-b0':
            # EfficientNet-B0 for better accuracy-efficiency trade-off
            self.backbone = EfficientNetB0(
                width_coefficient=0.5,  # Reduce width for mobile
                depth_coefficient=0.5,
                dropout_rate=0.2
            )
        
        # Projection head for embedding
        self.projection = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.PReLU()
        )
        
        # Feature normalization for better matching
        self.l2_norm = nn.functional.normalize
    
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] for single image or [B, T, 3, H, W] for video
        Returns:
            features: [B, embedding_dim] or [B, T, embedding_dim]
        """
        if len(x.shape) == 5:  # Video input
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)
            features = self.backbone(x)
            features = self.projection(features)
            features = features.reshape(B, T, -1)
        else:  # Image input
            features = self.backbone(x)
            features = self.projection(features)
        
        # L2 normalization for cosine similarity
        features = self.l2_norm(features, p=2, dim=-1)
        return features

class BottleneckBlock(nn.Module):
    """Inverted residual block for MobileFaceNet"""
    def __init__(self, in_channels, out_channels, stride, expansion):
        super().__init__()
        hidden_dim = in_channels * expansion
        
        self.use_residual = stride == 1 and in_channels == out_channels
        
        self.conv = nn.Sequential(
            # Pointwise expansion
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.PReLU(),
            
            # Depthwise convolution
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.PReLU(),
            
            # Pointwise projection
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)