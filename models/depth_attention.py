import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthAttentionModule(nn.Module):
    """
    Uses depth map to guide attention on facial features
    Emphasizes 3D structure, suppresses flat regions
    """
    
    def __init__(self, feature_dim=256, depth_dim=32):
        super().__init__()
        
        # Depth feature extractor
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, depth_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(depth_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))  # Match feature map size
        )
        
        # Attention generation network
        self.attention_generator = nn.Sequential(
            nn.Conv2d(depth_dim, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Depth quality assessment
        self.depth_quality = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(depth_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Spatial transformer for feature alignment
        self.spatial_transformer = SpatialTransformer()
    
    def forward(self, features, depth_map):
        """
        Args:
            features: [B, T, D, H, W] - spatial features from face encoder
            depth_map: [B, T, 1, H', W'] - estimated depth maps
        Returns:
            attended_features: [B, T, D] - depth-attended features
        """
        B, T, D, H, W = features.shape
        
        # Process each frame
        attended_features_list = []
        
        for t in range(T):
            feat_t = features[:, t]  # [B, D, H, W]
            depth_t = depth_map[:, t]  # [B, 1, H', W']
            
            # Encode depth information
            depth_encoded = self.depth_encoder(depth_t)  # [B, depth_dim, 7, 7]
            
            # Generate spatial attention map
            attention_map = self.attention_generator(depth_encoded)  # [B, 1, 7, 7]
            
            # Resize attention to match feature size
            if attention_map.shape[-2:] != feat_t.shape[-2:]:
                attention_map = F.interpolate(
                    attention_map, 
                    size=feat_t.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            # Apply attention
            attended_feat = feat_t * attention_map.expand_as(feat_t)
            
            # Assess depth quality for weighted combination
            depth_weight = self.depth_quality(depth_encoded)  # [B, 1]
            
            # Weighted combination of attended and original features
            final_feat = depth_weight.unsqueeze(-1).unsqueeze(-1) * attended_feat + \
                        (1 - depth_weight.unsqueeze(-1).unsqueeze(-1)) * feat_t
            
            # Global pooling
            final_feat = F.adaptive_avg_pool2d(final_feat, (1, 1)).squeeze(-1).squeeze(-1)
            
            attended_features_list.append(final_feat)
        
        # Stack temporal dimension
        attended_features = torch.stack(attended_features_list, dim=1)  # [B, T, D]
        
        return attended_features