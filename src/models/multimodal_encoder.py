"""
Multimodal Encoder - Step 3: Model Architecture & Training Objective Design

This module implements the Multimodal Encoder components:
1. Vision Transformer (ViT) Branch - Processes screenshot patches with masked self-attention
2. Structure Transformer Branch - Processes HTML tokens with hierarchical attention  
3. Token Fusion Module - Combines modalities using cross-attention and sparse fusion

Architecture specifications from instruction:
- Layers: 12
- Hidden dim: 768
- Heads: 12
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any
import warnings


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer inputs"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings"""
        return x + self.pe[:x.size(0), :]


class MaskedMultiHeadAttention(nn.Module):
    """Masked Multi-Head Attention for MaskDiT (reduces computation by 50%)"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, mask_ratio: float = 0.5):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.mask_ratio = mask_ratio
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None, training: bool = True) -> torch.Tensor:
        """
        Forward pass with optional masking for training efficiency
        
        Args:
            query, key, value: Input tensors [batch, seq_len, d_model]
            mask: Optional attention mask
            training: Whether in training mode (enables masking)
        """
        batch_size, seq_len, d_model = query.size()
        original_seq_len = seq_len
        
        # Apply masking during training for MaskDiT
        ids_keep = None
        if training and self.mask_ratio > 0:
            # Randomly mask tokens to reduce computation
            num_keep = int(seq_len * (1 - self.mask_ratio))
            noise = torch.rand(batch_size, seq_len, device=query.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_keep = ids_shuffle[:, :num_keep]
            
            # Keep only unmasked tokens
            query = torch.gather(query, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, d_model))
            key = torch.gather(key, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, d_model))
            value = torch.gather(value, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, d_model))
            seq_len = num_keep
        
        # Compute attention
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None and not training:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        output = self.w_o(attn_output)
        
        # If masking was applied, restore original sequence length with zeros
        if training and self.mask_ratio > 0 and ids_keep is not None:
            full_output = torch.zeros(batch_size, original_seq_len, d_model, device=output.device)
            full_output.scatter_(1, ids_keep.unsqueeze(-1).repeat(1, 1, d_model), output)
            output = full_output
        
        return output


class HierarchicalAttention(nn.Module):
    """Hierarchical Multi-Head Attention for preserving DOM relationships"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, hierarchy_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Forward pass with hierarchical attention
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            hierarchy_mask: Optional mask encoding DOM hierarchy relationships
        """
        # Self-attention with hierarchical bias
        attn_output, _ = self.attention(x, x, x, attn_mask=hierarchy_mask)
        
        # Residual connection and layer norm
        output = self.layer_norm(x + attn_output)
        
        return output


class TransformerBlock(nn.Module):
    """Standard Transformer block with feed-forward network"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int = None, dropout: float = 0.1, 
                 attention_type: str = "standard"):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
            
        self.attention_type = attention_type
        
        # Choose attention mechanism
        if attention_type == "masked":
            self.attention = MaskedMultiHeadAttention(d_model, num_heads, dropout)
        elif attention_type == "hierarchical":
            self.attention = HierarchicalAttention(d_model, num_heads, dropout)
        else:
            self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through transformer block"""
        
        # Attention block
        if self.attention_type == "masked":
            attn_output = self.attention(x, x, x, mask, training=self.training)
            x = self.norm1(x + attn_output)
        elif self.attention_type == "hierarchical":
            attn_output = self.attention(x, mask)
            x = self.norm1(x + attn_output - x)  # Since hierarchical attention already does residual + norm
        else:
            attn_output, _ = self.attention(x, x, x, attn_mask=mask)
            x = self.norm1(x + attn_output)
        
        # Feed-forward block
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) Branch for processing screenshot patches
    Uses masked self-attention (MaskDiT) to reduce computation
    
    Input: [batch, num_patches, patch_embed_dim]
    Output: [batch, num_patches, d_model]
    """
    
    def __init__(self, patch_embed_dim: int = 768, d_model: int = 768, num_layers: int = 12, 
                 num_heads: int = 12, dropout: float = 0.1, mask_ratio: float = 0.5):
        super().__init__()
        
        self.d_model = d_model
        self.patch_embed_dim = patch_embed_dim
        
        # Patch embedding projection
        self.patch_projection = nn.Linear(patch_embed_dim, d_model) if patch_embed_dim != d_model else nn.Identity()
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=1024)  # Support up to 1024 patches
        
        # 2D position embedding layer
        self.pos_embed_2d = nn.Linear(2, d_model)
        
        # Transformer blocks with masked attention
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dropout=dropout, attention_type="masked")
            for _ in range(num_layers)
        ])
        
        # Output layer norm
        self.norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, patch_embeddings: torch.Tensor, 
                patch_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through Vision Transformer
        
        Args:
            patch_embeddings: Patch embeddings [batch, num_patches, patch_embed_dim]
            patch_positions: Optional 2D position encodings [batch, num_patches, 2]
            
        Returns:
            Vision features [batch, num_patches, d_model]
        """
        batch_size, num_patches, _ = patch_embeddings.shape
        
        # Project patch embeddings to model dimension
        x = self.patch_projection(patch_embeddings)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # [num_patches, batch, d_model] for pos encoding
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # Back to [batch, num_patches, d_model]
        
        # Add 2D positional information if provided
        if patch_positions is not None:
            # Convert 2D positions to additional embeddings
            x = x + self.pos_embed_2d(patch_positions)
        
        x = self.dropout(x)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        return x


class StructureTransformer(nn.Module):
    """
    Structure Transformer Branch for processing HTML tokens
    Employs hierarchical attention to preserve DOM relationships
    
    Input: [batch, num_tokens, token_embed_dim] 
    Output: [batch, num_tokens, d_model]
    """
    
    def __init__(self, vocab_size: int, token_embed_dim: int = 768, d_model: int = 768, 
                 num_layers: int = 12, num_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.token_embed_dim = token_embed_dim
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, token_embed_dim)
        
        # Token embedding projection
        self.token_projection = nn.Linear(token_embed_dim, d_model) if token_embed_dim != d_model else nn.Identity()
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=512)  # Support up to 512 tokens
        
        # Hierarchy embedding for DOM relationships
        self.hierarchy_embedding = nn.Linear(2, d_model)  # depth, sibling_index
        
        # Transformer blocks with hierarchical attention
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dropout=dropout, attention_type="hierarchical")
            for _ in range(num_layers)
        ])
        
        # Output layer norm
        self.norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, token_ids: torch.Tensor, 
                hierarchy_embeddings: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through Structure Transformer
        
        Args:
            token_ids: Token IDs [batch, num_tokens]
            hierarchy_embeddings: Hierarchy info [batch, num_tokens, 2] (depth, sibling_index)
            attention_mask: Attention mask [batch, num_tokens]
            
        Returns:
            Structure features [batch, num_tokens, d_model]
        """
        batch_size, num_tokens = token_ids.shape
        
        # Token embeddings
        x = self.token_embedding(token_ids)  # [batch, num_tokens, token_embed_dim]
        
        # Project to model dimension
        x = self.token_projection(x)  # [batch, num_tokens, d_model]
        
        # Add positional encoding
        x = x.transpose(0, 1)  # [num_tokens, batch, d_model]
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # Back to [batch, num_tokens, d_model]
        
        # Add hierarchy embeddings if provided
        if hierarchy_embeddings is not None:
            hierarchy_emb = self.hierarchy_embedding(hierarchy_embeddings.float())
            x = x + hierarchy_emb
        
        x = self.dropout(x)
        
        # Create hierarchy mask for attention
        hierarchy_mask = None
        if attention_mask is not None:
            # Convert attention mask to 2D format for multi-head attention
            # attention_mask: [batch, num_tokens] - True for valid tokens, False for padded
            # We need a 2D mask: [num_tokens, num_tokens] where invalid positions are -inf
            
            # First convert to float and invert (True=valid becomes False=not masked)
            mask_2d = attention_mask.unsqueeze(-1) & attention_mask.unsqueeze(1)  # [batch, num_tokens, num_tokens]
            
            # Use only the first batch item for the 2D mask (assuming all batches have same pattern)
            hierarchy_mask = mask_2d[0]  # [num_tokens, num_tokens]
            
            # Convert to additive mask: False positions get -inf
            hierarchy_mask = hierarchy_mask.float()
            hierarchy_mask = hierarchy_mask.masked_fill(~hierarchy_mask.bool(), float('-inf'))
            hierarchy_mask = hierarchy_mask.masked_fill(hierarchy_mask.bool(), 0.0)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, hierarchy_mask)
        
        x = self.norm(x)
        
        return x


class CrossAttention(nn.Module):
    """Cross-attention module for token fusion"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cross-attention forward pass"""
        attn_output, attn_weights = self.attention(query, key, value, attn_mask=attn_mask)
        output = self.norm(query + attn_output)
        return output, attn_weights


class SparseFusion(nn.Module):
    """Sparse fusion module to prune redundant tokens"""
    
    def __init__(self, d_model: int, sparsity_ratio: float = 0.3):
        super().__init__()
        self.sparsity_ratio = sparsity_ratio
        self.importance_scorer = nn.Linear(d_model, 1)
        
    def forward(self, fused_tokens: torch.Tensor) -> torch.Tensor:
        """
        Apply sparse fusion to prune redundant tokens
        
        Args:
            fused_tokens: Fused token representations [batch, num_tokens, d_model]
            
        Returns:
            Pruned tokens [batch, num_pruned_tokens, d_model]
        """
        batch_size, num_tokens, d_model = fused_tokens.shape
        
        # Compute importance scores
        importance_scores = self.importance_scorer(fused_tokens).squeeze(-1)  # [batch, num_tokens]
        
        # Determine number of tokens to keep
        num_keep = int(num_tokens * (1 - self.sparsity_ratio))
        
        # Select top-k most important tokens
        _, top_indices = torch.topk(importance_scores, num_keep, dim=1)
        
        # Gather selected tokens
        expanded_indices = top_indices.unsqueeze(-1).expand(-1, -1, d_model)
        pruned_tokens = torch.gather(fused_tokens, 1, expanded_indices)
        
        return pruned_tokens


class TokenFusionModule(nn.Module):
    """
    Token Fusion Module - Combines vision and structure modalities
    Uses cross-attention between vision/structure tokens and sparse fusion
    
    Output: Unified [batch, num_fused_tokens, d_model]
    """
    
    def __init__(self, d_model: int = 768, num_heads: int = 12, dropout: float = 0.1, 
                 sparsity_ratio: float = 0.3):
        super().__init__()
        
        self.d_model = d_model
        
        # Cross-attention layers
        self.vision_to_structure = CrossAttention(d_model, num_heads, dropout)
        self.structure_to_vision = CrossAttention(d_model, num_heads, dropout)
        
        # Self-attention for final fusion
        self.fusion_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        
        # Sparse fusion module
        self.sparse_fusion = SparseFusion(d_model, sparsity_ratio)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward for final processing
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, vision_features: torch.Tensor, structure_features: torch.Tensor,
                vision_mask: Optional[torch.Tensor] = None, 
                structure_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Fuse vision and structure features using cross-attention
        
        Args:
            vision_features: Vision features [batch, num_patches, d_model]
            structure_features: Structure features [batch, num_tokens, d_model]
            vision_mask: Optional mask for vision features
            structure_mask: Optional mask for structure features
            
        Returns:
            Dictionary containing fused features and attention weights
        """
        
        # Cross-attention: Vision attends to Structure
        # Don't pass structure_mask as it has different dimensions than vision
        vision_attended, v2s_weights = self.vision_to_structure(
            vision_features, structure_features, structure_features, attn_mask=None
        )
        
        # Cross-attention: Structure attends to Vision  
        # Don't pass vision_mask as it has different dimensions than structure
        structure_attended, s2v_weights = self.structure_to_vision(
            structure_features, vision_features, vision_features, attn_mask=None
        )
        
        # Concatenate attended features
        fused_features = torch.cat([vision_attended, structure_attended], dim=1)
        
        # Self-attention on fused features
        fused_attended, fusion_weights = self.fusion_attention(
            fused_features, fused_features, fused_features
        )
        fused_features = self.norm1(fused_features + fused_attended)
        
        # Feed-forward processing
        ff_output = self.feed_forward(fused_features)
        fused_features = self.norm2(fused_features + ff_output)
        
        # Apply sparse fusion to prune redundant tokens
        pruned_features = self.sparse_fusion(fused_features)
        
        return {
            'fused_features': pruned_features,
            'full_fused_features': fused_features,
            'vision_attended': vision_attended,
            'structure_attended': structure_attended,
            'attention_weights': {
                'vision_to_structure': v2s_weights,
                'structure_to_vision': s2v_weights,
                'fusion': fusion_weights
            }
        }


class MultimodalEncoder(nn.Module):
    """
    Complete Multimodal Encoder combining Vision Transformer, Structure Transformer, 
    and Token Fusion Module
    
    Architecture specifications:
    - Layers: 12
    - Hidden dim: 768  
    - Heads: 12
    """
    
    def __init__(self, patch_embed_dim: int = 768, structure_vocab_size: int = 1000,
                 d_model: int = 768, num_layers: int = 12, num_heads: int = 12,
                 dropout: float = 0.1, mask_ratio: float = 0.5, sparsity_ratio: float = 0.3):
        super().__init__()
        
        self.d_model = d_model
        
        # Vision Transformer Branch
        self.vision_transformer = VisionTransformer(
            patch_embed_dim=patch_embed_dim,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            mask_ratio=mask_ratio
        )
        
        # Structure Transformer Branch
        self.structure_transformer = StructureTransformer(
            vocab_size=structure_vocab_size,
            token_embed_dim=d_model,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Token Fusion Module
        self.token_fusion = TokenFusionModule(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            sparsity_ratio=sparsity_ratio
        )
        
    def forward(self, patch_embeddings: torch.Tensor, patch_positions: Optional[torch.Tensor] = None,
                token_ids: torch.Tensor = None, hierarchy_embeddings: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through complete multimodal encoder
        
        Args:
            patch_embeddings: Patch embeddings [batch, num_patches, patch_embed_dim]
            patch_positions: Patch positions [batch, num_patches, 2]
            token_ids: Structure token IDs [batch, num_tokens]
            hierarchy_embeddings: Hierarchy info [batch, num_tokens, 2]
            attention_mask: Structure attention mask [batch, num_tokens]
            
        Returns:
            Dictionary containing multimodal features and intermediate outputs
        """
        
        # Process vision features
        vision_features = self.vision_transformer(patch_embeddings, patch_positions)
        
        # Process structure features
        structure_features = self.structure_transformer(
            token_ids, hierarchy_embeddings, attention_mask
        )
        
        # Fuse modalities
        fusion_output = self.token_fusion(
            vision_features, structure_features,
            vision_mask=None,  # Vision typically doesn't need masking
            structure_mask=attention_mask
        )
        
        return {
            'multimodal_features': fusion_output['fused_features'],
            'vision_features': vision_features,
            'structure_features': structure_features,
            'fusion_details': fusion_output
        }


# Utility functions for model configuration
def create_multimodal_encoder_config():
    """Create default configuration for multimodal encoder"""
    return {
        'patch_embed_dim': 768,
        'structure_vocab_size': 1000,
        'd_model': 768,
        'num_layers': 12,
        'num_heads': 12,
        'dropout': 0.1,
        'mask_ratio': 0.5,
        'sparsity_ratio': 0.3
    }


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
