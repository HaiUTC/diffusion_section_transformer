"""
Diffusion Decoder - Step 3: Model Architecture Implementation

This module implements:
- Conditional Denoising Transformer
- LayoutDenoiser class with joint cross-self attention
- Output heads for element and props prediction
- Timestep conditioning and noise prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Optional, Tuple
from .layout_embedding import LayoutEmbedding


class AdaptiveLayerNorm(nn.Module):
    """Adaptive Layer Normalization for timestep conditioning (AdaLN)"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        
    def forward(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive layer normalization
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            scale: Scale parameter [batch, d_model]
            shift: Shift parameter [batch, d_model]
        """
        # Apply layer norm without learnable parameters
        x = self.layer_norm(x)
        
        # Apply adaptive modulation
        scale = scale.unsqueeze(1)  # [batch, 1, d_model]
        shift = shift.unsqueeze(1)  # [batch, 1, d_model]
        
        return x * (1 + scale) + shift


class JointCrossSelfAttention(nn.Module):
    """Joint cross-self attention mechanism for encoder-decoder interaction"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Self-attention projections
        self.w_q_self = nn.Linear(d_model, d_model)
        self.w_k_self = nn.Linear(d_model, d_model)
        self.w_v_self = nn.Linear(d_model, d_model)
        
        # Cross-attention projections (query from decoder, key/value from encoder)
        self.w_q_cross = nn.Linear(d_model, d_model)
        self.w_k_cross = nn.Linear(d_model, d_model)
        self.w_v_cross = nn.Linear(d_model, d_model)
        
        # Output projections
        self.w_o_self = nn.Linear(d_model, d_model)
        self.w_o_cross = nn.Linear(d_model, d_model)
        
        # Combination layer
        self.combine = nn.Linear(d_model * 2, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, decoder_hidden: torch.Tensor, encoder_hidden: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                cross_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Joint cross-self attention forward pass
        
        Args:
            decoder_hidden: Decoder hidden states [batch, dec_seq_len, d_model]
            encoder_hidden: Encoder hidden states [batch, enc_seq_len, d_model]
            attention_mask: Self-attention mask [batch, dec_seq_len, dec_seq_len]
            cross_mask: Cross-attention mask [batch, dec_seq_len, enc_seq_len]
        """
        batch_size, dec_seq_len, _ = decoder_hidden.shape
        _, enc_seq_len, _ = encoder_hidden.shape
        
        # Self-attention
        Q_self = self.w_q_self(decoder_hidden).view(batch_size, dec_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K_self = self.w_k_self(decoder_hidden).view(batch_size, dec_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V_self = self.w_v_self(decoder_hidden).view(batch_size, dec_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Self-attention scores
        scores_self = torch.matmul(Q_self, K_self.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if attention_mask is not None:
            # Properly expand mask for multi-head attention
            # attention_mask: [batch, seq_len, seq_len] -> [batch, 1, seq_len, seq_len]
            # to match scores_self: [batch, num_heads, seq_len, seq_len]
            expanded_mask = attention_mask.unsqueeze(1)
            scores_self = scores_self.masked_fill(expanded_mask == 0, -1e9)
        
        attn_weights_self = F.softmax(scores_self, dim=-1)
        attn_weights_self = self.dropout(attn_weights_self)
        
        attn_output_self = torch.matmul(attn_weights_self, V_self)
        attn_output_self = attn_output_self.transpose(1, 2).contiguous().view(batch_size, dec_seq_len, self.d_model)
        attn_output_self = self.w_o_self(attn_output_self)
        
        # Cross-attention
        Q_cross = self.w_q_cross(decoder_hidden).view(batch_size, dec_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K_cross = self.w_k_cross(encoder_hidden).view(batch_size, enc_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V_cross = self.w_v_cross(encoder_hidden).view(batch_size, enc_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Cross-attention scores
        scores_cross = torch.matmul(Q_cross, K_cross.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if cross_mask is not None:
            # Properly expand mask for multi-head attention
            # cross_mask: [batch, dec_seq_len, enc_seq_len] -> [batch, 1, dec_seq_len, enc_seq_len]
            # to match scores_cross: [batch, num_heads, dec_seq_len, enc_seq_len]
            expanded_cross_mask = cross_mask.unsqueeze(1)
            scores_cross = scores_cross.masked_fill(expanded_cross_mask == 0, -1e9)
        
        attn_weights_cross = F.softmax(scores_cross, dim=-1)
        attn_weights_cross = self.dropout(attn_weights_cross)
        
        attn_output_cross = torch.matmul(attn_weights_cross, V_cross)
        attn_output_cross = attn_output_cross.transpose(1, 2).contiguous().view(batch_size, dec_seq_len, self.d_model)
        attn_output_cross = self.w_o_cross(attn_output_cross)
        
        # Combine self and cross attention
        combined = torch.cat([attn_output_self, attn_output_cross], dim=-1)
        output = self.combine(combined)
        
        return output


class LayoutDenoiserBlock(nn.Module):
    """Transformer block with joint cross-self attention for layout denoising"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        
        # Joint attention mechanism
        self.attention = JointCrossSelfAttention(d_model, num_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Adaptive layer normalization for timestep conditioning
        self.norm1 = AdaptiveLayerNorm(d_model)
        self.norm2 = AdaptiveLayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, encoder_hidden: torch.Tensor,
                timestep_scale: torch.Tensor, timestep_shift: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                cross_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through layout denoiser block"""
        
        # Joint attention with residual connection and adaptive norm
        attn_output = self.attention(x, encoder_hidden, attention_mask, cross_mask)
        x = self.norm1(x + attn_output, timestep_scale, timestep_shift)
        
        # Feed-forward with residual connection and adaptive norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output, timestep_scale, timestep_shift)
        
        return x


class LayoutDenoiser(nn.Module):
    """
    Layout Denoiser with joint cross-self attention
    
    Architecture:
    - Input: Noised layout tokens + timestep embeddings
    - Uses encoder-decoder attention with fused multimodal tokens
    - 12 transformer blocks with joint cross-self attention
    """
    
    def __init__(self, d_model: int = 768, num_layers: int = 12, num_heads: int = 12,
                 element_vocab_size: int = 100, property_vocab_size: int = 50,
                 max_elements: int = 50, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.max_elements = max_elements
        
        # Layout embedding module
        self.embed = LayoutEmbedding(
            d_model=d_model,
            element_vocab_size=element_vocab_size,
            property_vocab_size=property_vocab_size,
            dropout=dropout
        )
        
        # Transformer blocks with joint cross-self attention
        self.blocks = nn.ModuleList([
            LayoutDenoiserBlock(d_model, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Output layer normalization
        self.norm_out = nn.LayerNorm(d_model)
        
    def forward(self, noised_layout: torch.Tensor, timesteps: torch.Tensor,
                encoder_features: torch.Tensor,
                layout_mask: Optional[torch.Tensor] = None,
                encoder_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through layout denoiser
        
        Args:
            noised_layout: Noised layout tokens [batch, seq_len]
            timesteps: Diffusion timesteps [batch]
            encoder_features: Multimodal encoder features [batch, enc_seq_len, d_model]
            layout_mask: Layout attention mask [batch, seq_len, seq_len]
            encoder_mask: Encoder attention mask [batch, seq_len, enc_seq_len]
        """
        batch_size, seq_len = noised_layout.shape
        enc_seq_len = encoder_features.size(1)
        
        # Get layout embeddings with timestep conditioning
        embeddings = self.embed(
            layout_tokens=noised_layout,
            timesteps=timesteps
        )
        
        x = embeddings['token_embeddings']
        timestep_scale = embeddings['timestep_scale']
        timestep_shift = embeddings['timestep_shift']
        
        # Create attention masks with correct dimensions
        if layout_mask is None:
            # Self-attention mask: [batch, seq_len, seq_len]
            layout_mask = torch.ones(batch_size, seq_len, seq_len, device=x.device, dtype=torch.bool)
        
        if encoder_mask is None:
            # Cross-attention mask: [batch, seq_len, enc_seq_len]
            encoder_mask = torch.ones(batch_size, seq_len, enc_seq_len, device=x.device, dtype=torch.bool)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(
                x, encoder_features,
                timestep_scale, timestep_shift,
                layout_mask, encoder_mask
            )
        
        # Final normalization
        x = self.norm_out(x)
        
        return x


class DiffusionDecoder(nn.Module):
    """
    Complete Diffusion Decoder for layout generation
    
    Components:
    - LayoutDenoiser for conditional denoising
    - Output heads for element and props prediction
    - Noise prediction for diffusion training
    """
    
    def __init__(self, d_model: int = 768, num_layers: int = 12, num_heads: int = 12,
                 element_vocab_size: int = 100, property_vocab_size: int = 50,
                 max_elements: int = 50, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.element_vocab_size = element_vocab_size
        self.property_vocab_size = property_vocab_size
        self.max_elements = max_elements
        
        # Layout denoiser
        self.denoiser = LayoutDenoiser(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            element_vocab_size=element_vocab_size,
            property_vocab_size=property_vocab_size,
            max_elements=max_elements,
            dropout=dropout
        )
        
        # Output heads
        self.element_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, element_vocab_size + property_vocab_size)  # Combined vocab
        )
        
        # Props prediction heads
        self.props_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3)  # bi/bo/bv classifiers
        )
        
        # Noise prediction head (for diffusion training)
        self.noise_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)  # Predict noise in embedding space
        )
        
        # Geometric prediction head
        self.geometric_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 6)  # x, y, w, h, aspect_ratio, area
        )
        
    def forward(self, noised_layout: torch.Tensor, timesteps: torch.Tensor,
                encoder_features: torch.Tensor,
                layout_mask: Optional[torch.Tensor] = None,
                encoder_mask: Optional[torch.Tensor] = None,
                return_noise: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass through diffusion decoder
        
        Args:
            noised_layout: Noised layout tokens [batch, seq_len]
            timesteps: Diffusion timesteps [batch]
            encoder_features: Multimodal encoder features [batch, enc_seq_len, d_model]
            layout_mask: Layout attention mask [batch, seq_len]
            encoder_mask: Encoder attention mask [batch, enc_seq_len]
            return_noise: Whether to return noise prediction for training
            
        Returns:
            Dictionary containing predictions
        """
        # Denoise layout
        denoised_features = self.denoiser(
            noised_layout, timesteps, encoder_features,
            layout_mask, encoder_mask
        )
        
        # Generate predictions
        outputs = {}
        
        # Element predictions
        element_logits = self.element_head(denoised_features)
        outputs['element_logits'] = element_logits
        
        # Props predictions (background image/overlay/video)
        props_logits = self.props_head(denoised_features.mean(dim=1))  # Pool over sequence
        outputs['props_logits'] = props_logits
        
        # Geometric predictions
        geometric_pred = self.geometric_head(denoised_features)
        outputs['geometric_predictions'] = geometric_pred
        
        # Noise prediction for diffusion training
        if return_noise:
            noise_pred = self.noise_head(denoised_features)
            outputs['noise_prediction'] = noise_pred
        
        return outputs
    
    def sample(self, encoder_features: torch.Tensor, num_steps: int = 50,
               guidance_scale: float = 7.5) -> Dict[str, torch.Tensor]:
        """
        Sample layout using diffusion process
        
        Args:
            encoder_features: Multimodal encoder features [batch, enc_seq_len, d_model]
            num_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            
        Returns:
            Generated layout predictions
        """
        batch_size, enc_seq_len, _ = encoder_features.shape
        device = encoder_features.device
        
        # Start with pure noise
        seq_len = self.max_elements
        layout_tokens = torch.randint(
            0, self.element_vocab_size + self.property_vocab_size,
            (batch_size, seq_len), device=device, dtype=torch.long
        )
        
        # Denoising loop
        for step in range(num_steps):
            # Current timestep
            t = torch.full((batch_size,), step, device=device, dtype=torch.long)
            
            # Conditional prediction
            outputs_cond = self.forward(
                layout_tokens, t, encoder_features, return_noise=True
            )
            
            # Unconditional prediction (for classifier-free guidance)
            outputs_uncond = self.forward(
                layout_tokens, t, torch.zeros_like(encoder_features), return_noise=True
            )
            
            # Apply classifier-free guidance
            noise_pred = outputs_uncond['noise_prediction'] + guidance_scale * (
                outputs_cond['noise_prediction'] - outputs_uncond['noise_prediction']
            )
            
            # Update layout tokens (simplified denoising step)
            # Convert noise prediction to token updates
            element_logits = outputs_cond['element_logits']
            
            # Use element predictions for token updates (discrete diffusion)
            new_tokens = torch.argmax(element_logits, dim=-1)
            
            # Gradually transition from noise to predictions
            alpha_t = step / num_steps  # 0 to 1
            if alpha_t > 0.5:  # Start using predictions after halfway point
                layout_tokens = new_tokens
            
            # Ensure tokens are valid integers
            layout_tokens = layout_tokens.clamp(0, self.element_vocab_size + self.property_vocab_size - 1).long()
        
        # Final prediction
        final_outputs = self.forward(
            layout_tokens, torch.zeros(batch_size, device=device, dtype=torch.long),
            encoder_features, return_noise=False
        )
        
        return final_outputs


# Utility functions
def create_diffusion_decoder_config():
    """Create default configuration for diffusion decoder"""
    return {
        'd_model': 768,
        'num_layers': 12,
        'num_heads': 12,
        'element_vocab_size': 100,
        'property_vocab_size': 50,
        'max_elements': 50,
        'dropout': 0.1
    }


def count_diffusion_parameters(model: DiffusionDecoder) -> int:
    """Count the number of trainable parameters in diffusion decoder"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 