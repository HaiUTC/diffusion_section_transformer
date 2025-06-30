#!/usr/bin/env python3
"""
Multimodal Encoder Usage Example - Step 3: Model Architecture

This script demonstrates how to use the completed Multimodal Encoder
with the data processing pipeline from previous tasks.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from src.models import MultimodalEncoder, create_multimodal_encoder_config, count_parameters
from src.data.transforms import ImageTransforms, StructureTransforms
from PIL import Image
import numpy as np


def create_dummy_data():
    """Create dummy data for demonstration"""
    
    # Create dummy image (simulating a screenshot)
    dummy_image = Image.fromarray(np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8))
    
    # Create dummy structure data (simulating HTML structure tokens)
    structure_tokens = torch.randint(0, 100, (64,))  # 64 tokens from vocab of 100
    hierarchy_info = torch.randint(0, 5, (64, 2))    # depth and sibling index
    attention_mask = torch.ones(64, dtype=torch.bool)  # All tokens are valid
    
    return dummy_image, structure_tokens, hierarchy_info, attention_mask


def demonstrate_multimodal_encoder():
    """Demonstrate complete multimodal encoder workflow"""
    
    print("🚀 Multimodal Encoder Demonstration - Step 3")
    print("=" * 50)
    
    # Create dummy data
    print("\n📊 Creating demonstration data...")
    image, structure_tokens, hierarchy_info, attention_mask = create_dummy_data()
    print(f"✓ Image size: {image.size}")
    print(f"✓ Structure tokens: {structure_tokens.shape}")
    print(f"✓ Hierarchy info: {hierarchy_info.shape}")
    
    # Initialize transforms
    print("\n🔄 Initializing transforms...")
    image_transforms = ImageTransforms(target_size=512, patch_size=16)
    structure_transforms = StructureTransforms(max_sequence_length=128)
    
    # Process image
    print("\n🖼️ Processing image with Vision Transforms...")
    image_result = image_transforms(image)
    patches = image_result['patches'].unsqueeze(0)  # Add batch dimension
    patch_positions = image_result['patch_positions'].unsqueeze(0)  # Add batch dimension
    print(f"✓ Patches: {patches.shape}")
    print(f"✓ Patch positions: {patch_positions.shape}")
    
    # Process structure
    print("\n🌳 Processing structure with Structure Transforms...")
    # Add batch dimension for transforms
    structure_result = structure_transforms(
        structure_tokens, 
        hierarchy_info, 
        vocab_size=100
    )
    processed_tokens = structure_result['tokens'].unsqueeze(0)  # Add batch dimension
    processed_hierarchy = hierarchy_info.unsqueeze(0)  # Use original hierarchy
    processed_mask = attention_mask.unsqueeze(0)
    print(f"✓ Processed tokens: {processed_tokens.shape}")
    print(f"✓ Processed hierarchy: {processed_hierarchy.shape}")
    print(f"✓ Attention mask: {processed_mask.shape}")
    
    # Initialize multimodal encoder
    print("\n🧠 Initializing Multimodal Encoder...")
    config = create_multimodal_encoder_config()
    config['structure_vocab_size'] = 100  # Match our dummy vocab
    config['num_layers'] = 6  # Smaller for demo
    
    encoder = MultimodalEncoder(**config)
    
    # Count parameters
    total_params = count_parameters(encoder)
    memory_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
    print(f"✓ Model parameters: {total_params:,}")
    print(f"✓ Estimated memory: {memory_mb:.1f} MB")
    
    # Forward pass
    print("\n🔀 Running multimodal encoding...")
    
    with torch.no_grad():  # No gradients needed for demo
        output = encoder(
            patch_embeddings=patches,
            patch_positions=patch_positions,
            token_ids=processed_tokens,
            hierarchy_embeddings=processed_hierarchy,
            attention_mask=processed_mask
        )
    
    # Display results
    print("\n📈 Multimodal Encoder Results:")
    print(f"✓ Vision features: {output['vision_features'].shape}")
    print(f"✓ Structure features: {output['structure_features'].shape}")
    print(f"✓ Multimodal features: {output['multimodal_features'].shape}")
    
    # Show fusion details
    fusion_details = output['fusion_details']
    print(f"✓ Vision attended: {fusion_details['vision_attended'].shape}")
    print(f"✓ Structure attended: {fusion_details['structure_attended'].shape}")
    print(f"✓ Full fused features: {fusion_details['full_fused_features'].shape}")
    
    # Attention analysis
    attention_weights = fusion_details['attention_weights']
    v2s_weights = attention_weights['vision_to_structure']
    s2v_weights = attention_weights['structure_to_vision']
    print(f"✓ Vision→Structure attention: {v2s_weights.shape}")
    print(f"✓ Structure→Vision attention: {s2v_weights.shape}")
    
    print("\n🎉 Multimodal Encoder demonstration completed successfully!")
    print("\n📋 Summary:")
    print("- ✅ Vision Transformer: Processed image patches with MaskDiT")
    print("- ✅ Structure Transformer: Processed HTML tokens with hierarchical attention")
    print("- ✅ Token Fusion: Combined modalities with cross-attention and sparse fusion")
    print("- ✅ End-to-End: Complete multimodal feature extraction ready for downstream tasks")
    

def demonstrate_config_options():
    """Demonstrate different configuration options"""
    
    print("\n⚙️ Configuration Options:")
    print("-" * 30)
    
    # Default config
    default_config = create_multimodal_encoder_config()
    print("Default Configuration:")
    for key, value in default_config.items():
        print(f"  {key}: {value}")
    
    # Custom configs for different use cases
    configs = {
        "Lightweight": {
            'd_model': 384,
            'num_layers': 6,
            'num_heads': 6,
            'mask_ratio': 0.7,
            'sparsity_ratio': 0.5
        },
        "Production": {
            'd_model': 768,
            'num_layers': 12,
            'num_heads': 12,
            'mask_ratio': 0.5,
            'sparsity_ratio': 0.3
        },
        "Research": {
            'd_model': 1024,
            'num_layers': 16,
            'num_heads': 16,
            'mask_ratio': 0.3,
            'sparsity_ratio': 0.2
        }
    }
    
    print("\nPre-configured Options:")
    for name, config in configs.items():
        estimated_params = estimate_parameters(config)
        print(f"\n{name} Config:")
        print(f"  Model size: ~{estimated_params/1e6:.1f}M parameters")
        print(f"  Hidden dim: {config['d_model']}")
        print(f"  Layers: {config['num_layers']}")
        print(f"  Attention heads: {config['num_heads']}")


def estimate_parameters(config):
    """Rough parameter estimation based on config"""
    d_model = config['d_model']
    num_layers = config['num_layers']
    
    # Rough estimation: each transformer layer has ~12 * d_model^2 parameters
    # Plus embeddings, projections, etc.
    layer_params = 12 * d_model * d_model
    total_params = num_layers * layer_params * 2  # Vision + Structure transformers
    total_params += d_model * 1000  # Embeddings
    total_params += d_model * d_model * 8  # Fusion module
    
    return total_params


if __name__ == "__main__":
    print("Multimodal Encoder Example - Step 3: Model Architecture")
    print("This example demonstrates the complete multimodal encoder implementation.")
    print("The encoder combines Vision and Structure transformers with Token Fusion.")
    
    try:
        demonstrate_multimodal_encoder()
        demonstrate_config_options()
        
        print("\n" + "="*60)
        print("🎯 Next Steps:")
        print("- Implement Diffusion Decoder for layout generation")
        print("- Add training objectives and loss functions")
        print("- Integrate aesthetic constraint modules")
        print("- Set up training pipeline")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        print("Make sure all dependencies are installed and paths are correct.")
        sys.exit(1)
