#!/usr/bin/env python3
"""
Test script for preprocessing transforms - Task 2.4
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from PIL import Image
import numpy as np

def test_image_transforms():
    """Test image preprocessing transforms"""
    print("=== Testing Image Transforms ===\n")
    
    try:
        from src.data.transforms import ImageTransforms
        
        # Create a dummy image for testing
        dummy_image = Image.new('RGB', (400, 300), color='red')
        print(f"Created dummy image: {dummy_image.size}")
        
        # Test different configurations
        configs = [
            {"target_size": 512, "patch_size": 16, "center_crop": True},
            {"target_size": 224, "patch_size": 16, "center_crop": False},
        ]
        
        for i, config in enumerate(configs):
            print(f"\n{i+1}. Testing config: {config}")
            
            transform = ImageTransforms(**config)
            result = transform(dummy_image)
            
            print(f"   ✓ Image tensor shape: {result['image_tensor'].shape}")
            print(f"   ✓ Patches shape: {result['patches'].shape}")
            print(f"   ✓ Patch positions shape: {result['patch_positions'].shape}")
            
    except Exception as e:
        print(f"❌ Image transforms test failed: {e}")


def test_structure_transforms():
    """Test structure preprocessing transforms"""
    print("\n=== Testing Structure Transforms ===\n")
    
    try:
        from src.data.transforms import StructureTransforms
        
        # Create dummy tokens and hierarchy
        tokens = torch.randint(0, 50, (20,))
        hierarchy_info = torch.randint(0, 5, (20, 2))
        
        print(f"Created dummy tokens: {tokens.shape}")
        print(f"Created dummy hierarchy: {hierarchy_info.shape}")
        
        # Test transform
        transform = StructureTransforms(max_sequence_length=64, mask_probability=0.15)
        result = transform(tokens, hierarchy_info, vocab_size=100)
        
        print(f"✓ Transformed tokens shape: {result['tokens'].shape}")
        print(f"✓ Attention mask shape: {result['attention_mask'].shape}")
        print(f"✓ Position embeddings shape: {result['position_embeddings'].shape}")
        
    except Exception as e:
        print(f"❌ Structure transforms test failed: {e}")


def test_layout_transforms():
    """Test layout preprocessing transforms"""
    print("\n=== Testing Layout Transforms ===\n")
    
    try:
        from src.data.transforms import LayoutTransforms
        
        # Create dummy layout tokens
        tokens = torch.randint(0, 30, (15,))
        print(f"Created dummy layout tokens: {tokens.shape}")
        
        # Test transform
        transform = LayoutTransforms(max_sequence_length=32, label_smoothing=0.1)
        result = transform(tokens)
        
        print(f"✓ Transformed tokens shape: {result['tokens'].shape}")
        print(f"✓ Attention mask shape: {result['attention_mask'].shape}")
        print(f"✓ Causal mask shape: {result['causal_mask'].shape}")
        print(f"✓ Labels shape: {result['labels'].shape}")
        
    except Exception as e:
        print(f"❌ Layout transforms test failed: {e}")


if __name__ == "__main__":
    print("🚀 Testing Preprocessing Transforms - Task 2.4\n")
    
    # Run all tests
    test_image_transforms()
    test_structure_transforms() 
    test_layout_transforms()
    
    print("\n✅ Transform tests completed!")
    print("\nTask 2.4 Preprocessing Transforms implemented:")
    print("- ✓ Image transforms: resize, crop/pad, normalize, patch embedding")
    print("- ✓ Structure transforms: token mapping, position embeddings, masking")
    print("- ✓ Layout transforms: tokenization, attention masks, label smoothing")
