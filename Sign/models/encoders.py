import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F

# Constants for frame dimensions
FRAME_HEIGHT = 160
FRAME_WIDTH = 160

class VisualEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        # Load pretrained EfficientNet
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        
        # Get the number of features from the backbone
        self.backbone_features = self.backbone._fc.in_features
        
        # Remove the original classifier
        self.backbone._fc = nn.Identity()
        
        # Add temporal reduction layers
        self.temporal_reduction = nn.Sequential(
            nn.Linear(self.backbone_features, output_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    
    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.shape
        
        # Reshape for EfficientNet
        x = x.view(-1, channels, height, width)
        
        # Extract visual features
        features = self.backbone.extract_features(x)
        
        # Global average pooling
        features = F.adaptive_avg_pool2d(features, 1)
        features = features.squeeze(-1).squeeze(-1)
        
        # Reshape back to include temporal dimension
        features = features.view(batch_size, num_frames, -1)
        
        # Reduce temporal dimension
        features = self.temporal_reduction(features)
        
        return features

class GestureEncoder(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()
        
        # Simple CNN for gesture feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Calculate feature size after CNN
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, FRAME_HEIGHT, FRAME_WIDTH)
            feature_size = self.feature_extractor(dummy_input).view(-1).shape[0]
        
        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
    
    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Reshape for CNN processing
        x = x.view(-1, 3, FRAME_HEIGHT, FRAME_WIDTH)
        
        # Extract features using CNN
        features = self.feature_extractor(x)
        features = features.view(batch_size, seq_len, -1)
        
        # Process through LSTM
        output, _ = self.lstm(features)
        return output 