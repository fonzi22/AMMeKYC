import torch
import torch.nn as nn

class DifferencingModule(nn.Module):
    """
    Computes multi-scale differences between reference and video features
    to detect identity inconsistencies and presentation attacks
    """
    
    def __init__(self, input_dim=256, output_dim=128):
        super().__init__()
        
        # Multiple differencing strategies
        self.diff_types = ['subtract', 'multiply', 'cosine', 'learned']
        
        # Learned differencing with attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=4,
            dropout=0.1
        )
        
        # Feature fusion for different difference types
        self.fusion_network = nn.Sequential(
            nn.Linear(input_dim * len(self.diff_types), 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, output_dim),
            nn.BatchNorm1d(output_dim)
        )
        
        # Learnable temperature for similarity scaling
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)
    
    def forward(self, ref_features, video_features):
        """
        Args:
            ref_features: [B, D] - features from reference image
            video_features: [B, T, D] - features from video frames
        Returns:
            diff_features: [B, T, output_dim] - difference features
        """
        B, T, D = video_features.shape
        
        # Expand reference features to match video
        ref_expanded = ref_features.unsqueeze(1).expand(B, T, D)
        
        differences = []
        
        # 1. Element-wise subtraction (L1 distance)
        diff_subtract = torch.abs(video_features - ref_expanded)
        differences.append(diff_subtract)
        
        # 2. Element-wise multiplication (correlation)
        diff_multiply = video_features * ref_expanded
        differences.append(diff_multiply)
        
        # 3. Cosine similarity (expanded)
        cosine_sim = F.cosine_similarity(video_features, ref_expanded, dim=-1)
        cosine_features = cosine_sim.unsqueeze(-1).expand(B, T, D)
        differences.append(cosine_features)
        
        # 4. Learned cross-attention difference
        video_flat = video_features.reshape(B * T, 1, D).transpose(0, 1)
        ref_flat = ref_expanded.reshape(B * T, 1, D).transpose(0, 1)
        
        attn_diff, _ = self.cross_attention(
            query=video_flat,
            key=ref_flat,
            value=ref_flat
        )
        attn_diff = attn_diff.transpose(0, 1).reshape(B, T, D)
        differences.append(attn_diff)
        
        # Concatenate all differences
        all_differences = torch.cat(differences, dim=-1)  # [B, T, D*4]
        
        # Reshape for batch norm
        all_diff_flat = all_differences.reshape(B * T, -1)
        
        # Fuse different difference types
        diff_features = self.fusion_network(all_diff_flat)
        diff_features = diff_features.reshape(B, T, -1)
        
        return diff_features