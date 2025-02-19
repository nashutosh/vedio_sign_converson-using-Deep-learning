import torch
import torch.nn as nn
from models.encoders import VisualEncoder, GestureEncoder
from models.decoders import TransformerDecoder

class SignLanguageTranslator(nn.Module):
    def __init__(
        self,
        gloss_vocab_size,
        text_vocab_size,
        d_model=512,
        nhead=8,
        num_decoder_layers=6
    ):
        super().__init__()
        
        # Encoders
        self.visual_encoder = VisualEncoder()
        self.gesture_encoder = GestureEncoder()
        
        # Feature fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(1024, d_model),  # 512 from each encoder
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Decoders
        self.gloss_decoder = TransformerDecoder(
            gloss_vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers
        )
        
        self.translation_decoder = TransformerDecoder(
            text_vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers
        )
        
    def forward(
        self,
        video_frames,
        gloss_targets=None,
        text_targets=None,
        gloss_mask=None,
        text_mask=None
    ):
        # Extract visual features
        visual_features = self.visual_encoder(video_frames)
        
        # Extract gesture features
        gesture_features = self.gesture_encoder(video_frames)
        
        # Fusion
        combined_features = torch.cat([visual_features, gesture_features], dim=-1)
        memory = self.fusion_layer(combined_features)
        
        # Decode gloss sequence
        gloss_output = self.gloss_decoder(
            gloss_targets,
            memory,
            tgt_mask=gloss_mask
        )
        
        # Decode text translation
        text_output = self.translation_decoder(
            text_targets,
            memory,
            tgt_mask=text_mask
        )
        
        return gloss_output, text_output 