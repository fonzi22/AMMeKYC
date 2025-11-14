class DualBranchDepthAwareFAS(nn.Module):
    """
    Dual-branch architecture for face anti-spoofing with reference image matching
    and pseudo-depth estimation for mobile deployment
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Shared face encoder
        self.face_encoder = SharedFaceEncoder(
            backbone=config.backbone,  # 'mobilefacenet' or 'efficientnet-b0'
            embedding_dim=config.embedding_dim  # 256
        )
        
        # Branch 1: Reference-Video Differencing
        self.differencing_module = DifferencingModule(
            input_dim=config.embedding_dim,
            output_dim=config.diff_dim  # 128
        )
        
        # Branch 2: Depth Estimation and Attention
        self.depth_estimator = LightweightDepthEstimator(
            input_channels=3,
            output_channels=1
        )
        self.depth_attention = DepthAttentionModule(
            feature_dim=config.embedding_dim,
            depth_dim=32
        )
        
        # Temporal modeling
        self.temporal_encoder = TemporalEncoder(
            input_dim=config.embedding_dim + config.diff_dim,
            hidden_dim=config.temporal_dim,  # 256
            num_layers=2
        )
        
        # Classification head
        self.classifier = ClassificationHead(
            input_dim=config.temporal_dim,
            num_classes=2  # real/fake or multi-class attacks
        )

video = torch.rand(1, 16, 3 , 112, 112)  # (Batch size, num frames, Channels, Height, Width)
face_image = torch.rand(1, 3 , 112, 112)  # (Batch size, Channels, Height, Width)
model = DualBranchDepthAwareFAS(config)
model(video, face_image) -> logits (real / fake), depth_map (Height, Width)
