"""
Model architecture implementations for Section Layout Generation

This package implements the Multimodal Diffusion Transformer (MDT) architecture
following the specifications in Step 3 of the instruction.

Complete implementation includes:
- Multimodal Encoder (Vision + Structure Transformers with Token Fusion)
- Layout Embedding (Geometric + Class + Timestep embeddings)
- Diffusion Decoder (Conditional Denoising Transformer)
- Aesthetic Constraint Module (Overlap, Alignment, Proportion constraints)
"""

# Multimodal Encoder components
from .multimodal_encoder import (
    MultimodalEncoder, VisionTransformer, StructureTransformer, TokenFusionModule,
    create_multimodal_encoder_config, count_parameters
)

# Layout Embedding components
from .layout_embedding import (
    LayoutEmbedding, GeometricEmbedding, ClassEmbedding, TimestepEmbedding,
    create_layout_embedding_config, normalize_geometric_features
)

# Diffusion Decoder components
from .diffusion_decoder import (
    DiffusionDecoder, LayoutDenoiser, LayoutDenoiserBlock, JointCrossSelfAttention,
    AdaptiveLayerNorm, create_diffusion_decoder_config, count_diffusion_parameters
)

# Aesthetic Constraint components
from .aesthetic_constraints import (
    AestheticConstraintModule, OverlapConstraint, AlignmentConstraint, 
    ProportionConstraint, ReadabilityConstraint, create_aesthetic_constraint_config,
    apply_aesthetic_guidance, box_iou
)

__all__ = [
    # Multimodal Encoder
    "MultimodalEncoder",
    "VisionTransformer", 
    "StructureTransformer",
    "TokenFusionModule",
    "create_multimodal_encoder_config",
    "count_parameters",
    
    # Layout Embedding
    "LayoutEmbedding",
    "GeometricEmbedding",
    "ClassEmbedding", 
    "TimestepEmbedding",
    "create_layout_embedding_config",
    "normalize_geometric_features",
    
    # Diffusion Decoder
    "DiffusionDecoder",
    "LayoutDenoiser",
    "LayoutDenoiserBlock",
    "JointCrossSelfAttention",
    "AdaptiveLayerNorm",
    "create_diffusion_decoder_config",
    "count_diffusion_parameters",
    
    # Aesthetic Constraints
    "AestheticConstraintModule",
    "OverlapConstraint",
    "AlignmentConstraint",
    "ProportionConstraint", 
    "ReadabilityConstraint",
    "create_aesthetic_constraint_config",
    "apply_aesthetic_guidance",
    "box_iou"
]
