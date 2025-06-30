"""
Layout Embedding - Step 3: Model Architecture Implementation

This module implements layout token embeddings with:
- Geometric embeddings (position, size, aspect ratio)
- Class embeddings (element types, properties)
- Position and size encodings for layout understanding
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Optional, Tuple, Any


class GeometricEmbedding(nn.Module):
    """Geometric embeddings for layout elements (position, size, aspect ratio)"""
    
    def __init__(self, d_model: int = 768, max_position: int = 1000):
        super().__init__()
        self.d_model = d_model
        
        # Position embeddings (x, y coordinates)
        self.pos_x_embedding = nn.Embedding(max_position, d_model // 4)
        self.pos_y_embedding = nn.Embedding(max_position, d_model // 4)
        
        # Size embeddings (width, height)
        self.width_embedding = nn.Embedding(max_position, d_model // 4)
        self.height_embedding = nn.Embedding(max_position, d_model // 4)
        
        # Continuous geometric features projection
        self.geometric_projection = nn.Linear(6, d_model)  # x,y,w,h,aspect_ratio,area
        
        # Combine discrete and continuous embeddings
        self.fusion_layer = nn.Linear(d_model * 2, d_model)
        
    def forward(self, geometric_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for geometric embeddings
        
        Args:
            geometric_features: [batch, num_elements, 6] containing [x, y, w, h, aspect_ratio, area]
            
        Returns:
            Geometric embeddings [batch, num_elements, d_model]
        """
        batch_size, num_elements, _ = geometric_features.shape
        
        # Extract geometric components (assuming normalized to [0, 999])
        x = geometric_features[:, :, 0].long().clamp(0, 999)
        y = geometric_features[:, :, 1].long().clamp(0, 999)
        w = geometric_features[:, :, 2].long().clamp(0, 999)
        h = geometric_features[:, :, 3].long().clamp(0, 999)
        
        # Discrete embeddings
        pos_x_emb = self.pos_x_embedding(x)      # [batch, num_elements, d_model//4]
        pos_y_emb = self.pos_y_embedding(y)      # [batch, num_elements, d_model//4]
        width_emb = self.width_embedding(w)      # [batch, num_elements, d_model//4]
        height_emb = self.height_embedding(h)    # [batch, num_elements, d_model//4]
        
        # Concatenate discrete embeddings
        discrete_emb = torch.cat([pos_x_emb, pos_y_emb, width_emb, height_emb], dim=-1)
        
        # Continuous embeddings
        continuous_emb = self.geometric_projection(geometric_features)
        
        # Combine discrete and continuous
        combined = torch.cat([discrete_emb, continuous_emb], dim=-1)
        output = self.fusion_layer(combined)
        
        return output


class ClassEmbedding(nn.Module):
    """Class embeddings for element types and properties"""
    
    def __init__(self, d_model: int = 768, element_vocab_size: int = 100, 
                 property_vocab_size: int = 50):
        super().__init__()
        self.d_model = d_model
        
        # Element type embeddings (section, heading, paragraph, etc.)
        self.element_embedding = nn.Embedding(element_vocab_size, d_model // 2)
        
        # Property embeddings (classes, attributes)
        self.property_embedding = nn.Embedding(property_vocab_size, d_model // 2)
        
        # Combination layer
        self.class_fusion = nn.Linear(d_model, d_model)
        
    def forward(self, element_ids: torch.Tensor, property_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for class embeddings
        
        Args:
            element_ids: Element type IDs [batch, num_elements]
            property_ids: Property IDs [batch, num_elements]
            
        Returns:
            Class embeddings [batch, num_elements, d_model]
        """
        # Element embeddings
        element_emb = self.element_embedding(element_ids)      # [batch, num_elements, d_model//2]
        
        # Property embeddings
        property_emb = self.property_embedding(property_ids)   # [batch, num_elements, d_model//2]
        
        # Combine embeddings
        combined = torch.cat([element_emb, property_emb], dim=-1)
        output = self.class_fusion(combined)
        
        return output


class TimestepEmbedding(nn.Module):
    """Timestep embeddings for diffusion process"""
    
    def __init__(self, d_model: int = 768, max_timesteps: int = 1000):
        super().__init__()
        self.d_model = d_model
        
        # Sinusoidal timestep embeddings
        self.timestep_projection = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model)
        )
        
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Create timestep embeddings using sinusoidal encoding
        
        Args:
            timesteps: Timestep values [batch] or [batch, 1]
            
        Returns:
            Timestep embeddings [batch, d_model]
        """
        if timesteps.dim() == 1:
            timesteps = timesteps.unsqueeze(-1)
        
        # Create sinusoidal embeddings
        half_dim = self.d_model // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps.float() * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # Project to model dimension
        emb = self.timestep_projection(emb)
        
        return emb


class LayoutEmbedding(nn.Module):
    """
    Complete Layout Embedding module combining geometric and class embeddings
    
    This module handles:
    - Geometric embeddings (position, size, aspect ratio)
    - Class embeddings (element types, properties)  
    - Timestep embeddings for diffusion process
    - Position and size encodings for layout understanding
    """
    
    def __init__(self, d_model: int = 768, element_vocab_size: int = 100,
                 property_vocab_size: int = 50, max_position: int = 1000,
                 max_timesteps: int = 1000, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Geometric embeddings
        self.geometric_embedding = GeometricEmbedding(d_model, max_position)
        
        # Class embeddings
        self.class_embedding = ClassEmbedding(d_model, element_vocab_size, property_vocab_size)
        
        # Timestep embeddings
        self.timestep_embedding = TimestepEmbedding(d_model, max_timesteps)
        
        # Layout token embeddings (for sequence modeling)
        self.token_embedding = nn.Embedding(element_vocab_size + property_vocab_size, d_model)
        
        # Positional encoding for sequence position
        self.register_buffer('pos_encoding', self._create_positional_encoding(1000, d_model))
        
        # Combination layers
        self.geometric_class_fusion = nn.Linear(d_model * 2, d_model)
        self.timestep_modulation = nn.Linear(d_model, d_model * 2)  # For AdaLN-style conditioning
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
        
    def forward(self, layout_tokens: Optional[torch.Tensor] = None,
                geometric_features: Optional[torch.Tensor] = None,
                element_ids: Optional[torch.Tensor] = None,
                property_ids: Optional[torch.Tensor] = None,
                timesteps: Optional[torch.Tensor] = None,
                sequence_positions: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through layout embedding
        
        Args:
            layout_tokens: Token IDs [batch, seq_len] (for sequence modeling)
            geometric_features: Geometric features [batch, num_elements, 6]
            element_ids: Element type IDs [batch, num_elements]
            property_ids: Property IDs [batch, num_elements]
            timesteps: Diffusion timesteps [batch]
            sequence_positions: Sequence positions [batch, seq_len]
            
        Returns:
            Dictionary containing various embeddings
        """
        batch_size = None
        embeddings = {}
        
        # Token embeddings for sequence modeling
        if layout_tokens is not None:
            batch_size, seq_len = layout_tokens.shape
            token_emb = self.token_embedding(layout_tokens)
            
            # Add positional encoding
            if sequence_positions is not None:
                pos_emb = self.pos_encoding[sequence_positions]
            else:
                pos_emb = self.pos_encoding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
            
            token_emb = token_emb + pos_emb
            embeddings['token_embeddings'] = token_emb
        
        # Geometric + Class embeddings for layout elements
        if geometric_features is not None and element_ids is not None and property_ids is not None:
            batch_size, num_elements = element_ids.shape
            
            # Get individual embeddings
            geom_emb = self.geometric_embedding(geometric_features)
            class_emb = self.class_embedding(element_ids, property_ids)
            
            # Combine geometric and class embeddings
            combined = torch.cat([geom_emb, class_emb], dim=-1)
            layout_emb = self.geometric_class_fusion(combined)
            
            embeddings['layout_embeddings'] = layout_emb
        
        # Timestep conditioning
        if timesteps is not None:
            if batch_size is None:
                batch_size = timesteps.shape[0]
            
            timestep_emb = self.timestep_embedding(timesteps)
            
            # AdaLN-style modulation parameters
            timestep_modulation = self.timestep_modulation(timestep_emb)
            scale, shift = timestep_modulation.chunk(2, dim=-1)
            
            embeddings['timestep_embeddings'] = timestep_emb
            embeddings['timestep_scale'] = scale
            embeddings['timestep_shift'] = shift
        
        # Apply layer norm and dropout to final embeddings
        for key in ['token_embeddings', 'layout_embeddings']:
            if key in embeddings:
                embeddings[key] = self.layer_norm(embeddings[key])
                embeddings[key] = self.dropout(embeddings[key])
        
        return embeddings


# Utility functions
def create_layout_embedding_config():
    """Create default configuration for layout embedding"""
    return {
        'd_model': 768,
        'element_vocab_size': 100,
        'property_vocab_size': 50,
        'max_position': 1000,
        'max_timesteps': 1000,
        'dropout': 0.1
    }


def normalize_geometric_features(x: float, y: float, width: float, height: float,
                                canvas_width: float = 1920, canvas_height: float = 1080) -> torch.Tensor:
    """
    Normalize geometric features to [0, 999] range for embedding lookup
    
    Args:
        x, y, width, height: Layout coordinates
        canvas_width, canvas_height: Canvas dimensions
        
    Returns:
        Normalized features tensor [6] containing [x, y, w, h, aspect_ratio, area]
    """
    # Normalize position and size
    norm_x = int((x / canvas_width) * 999)
    norm_y = int((y / canvas_height) * 999) 
    norm_w = int((width / canvas_width) * 999)
    norm_h = int((height / canvas_height) * 999)
    
    # Calculate derived features
    aspect_ratio = width / height if height > 0 else 1.0
    area = (width * height) / (canvas_width * canvas_height)
    
    # Normalize derived features
    norm_aspect = min(int(aspect_ratio * 100), 999)  # Cap at 10:1 aspect ratio
    norm_area = int(area * 999)
    
    return torch.tensor([norm_x, norm_y, norm_w, norm_h, norm_aspect, norm_area], dtype=torch.float32) 