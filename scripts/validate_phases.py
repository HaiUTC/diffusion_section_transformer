#!/usr/bin/env python3
"""
Phase Configuration Validation Script
Tests all phase configurations and validates the growth strategy implementation.
"""

import sys
import os
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ai_engine_configurable import ConfigurableSectionLayoutGenerator
from src.utils.config_loader import config_loader


def validate_phase_configs():
    """Validate all phase configurations can be loaded correctly."""
    
    print("🔍 VALIDATING PHASE CONFIGURATIONS")
    print("=" * 50)
    
    phases = ["phase1", "phase2", "phase3", "phase4"]
    configs = {}
    
    for phase in phases:
        try:
            # Load configuration
            config = config_loader.load_config(phase=phase)
            model_config = config_loader.get_model_config(phase=phase)
            training_config = config_loader.get_training_config(phase=phase)
            
            configs[phase] = {
                'config': config,
                'model': model_config,
                'training': training_config
            }
            
            print(f"✅ {phase.upper()}: Configuration loaded successfully")
            
        except Exception as e:
            print(f"❌ {phase.upper()}: Configuration failed - {e}")
            return False
    
    print("\n✅ All phase configurations validated!")
    return True


def validate_dataset_size_detection():
    """Test automatic phase detection based on dataset size."""
    
    print("\n🔍 VALIDATING DATASET SIZE DETECTION")
    print("=" * 50)
    
    test_cases = [
        (1500, "phase1"),    # User's current situation
        (2500, "phase2"),
        (5000, "phase3"),
        (15000, "phase4")
    ]
    
    for dataset_size, expected_phase in test_cases:
        try:
            detected_phase = config_loader.get_phase_by_dataset_size(dataset_size)
            
            if detected_phase == expected_phase:
                print(f"✅ Dataset size {dataset_size:,} → {detected_phase} (correct)")
            else:
                print(f"❌ Dataset size {dataset_size:,} → {detected_phase} (expected {expected_phase})")
                return False
                
        except Exception as e:
            print(f"❌ Dataset size {dataset_size:,} failed: {e}")
            return False
    
    print("\n✅ Dataset size detection working correctly!")
    return True


def validate_model_creation():
    """Test model creation for each phase."""
    
    print("\n🔍 VALIDATING MODEL CREATION")
    print("=" * 50)
    
    phases = ["phase1", "phase2", "phase3", "phase4"]
    models = {}
    
    for phase in phases:
        try:
            print(f"\n📝 Creating {phase.upper()} model...")
            
            # Create model
            model = ConfigurableSectionLayoutGenerator(phase=phase)
            models[phase] = model
            
            # Get model info
            info = model.get_model_info()
            
            print(f"   Parameters: {info['total_parameters']:,}")
            print(f"   Memory: {info['model_size_mb']:.1f} MB")
            print(f"   Phase: {info['phase']}")
            
            # Validate model can be put in eval mode
            model.eval()
            print(f"   ✅ Model evaluation mode: OK")
            
        except Exception as e:
            print(f"   ❌ Model creation failed: {e}")
            return False
    
    print(f"\n✅ All {len(models)} phase models created successfully!")
    return True


def test_phase1_inference():
    """Test Phase 1 model inference with mock data (user's current situation)."""
    
    print("\n🔍 TESTING PHASE 1 INFERENCE (Your Current Situation)")
    print("=" * 50)
    
    try:
        # Create Phase 1 model (user's current phase)
        model = ConfigurableSectionLayoutGenerator(dataset_size=1500)
        
        # Mock data similar to user's needs
        batch_size = 2
        screenshot = torch.randn(batch_size, 3, 224, 224)
        structure_tokens = torch.randint(0, 100, (batch_size, 128))
        
        print(f"📊 Input data:")
        print(f"   Screenshot shape: {screenshot.shape}")
        print(f"   Structure tokens shape: {structure_tokens.shape}")
        
        # Test inference
        model.eval()
        with torch.no_grad():
            outputs = model.generate_layout(
                screenshot=screenshot,
                structure_tokens=structure_tokens,
                num_steps=10  # Fast inference for testing
            )
        
        print(f"\n✅ Phase 1 inference successful!")
        print(f"   Generated elements: {len(outputs.elements)}")
        print(f"   Aesthetic score: {outputs.aesthetic_score:.2f}")
        print(f"   Constraint violations: {len(outputs.constraint_violations)}")
        print(f"   Confidence (avg): {outputs.confidence_scores.mean():.2f}")
        
        # Show sample elements
        print(f"\n📋 Sample generated elements:")
        for i, element in enumerate(outputs.elements[:3]):
            print(f"   {i+1}. {element['type']} at ({element['position']['x']:.2f}, {element['position']['y']:.2f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Phase 1 inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_upgrade():
    """Test model upgrade functionality."""
    
    print("\n🔍 TESTING MODEL UPGRADE")
    print("=" * 50)
    
    try:
        # Start with Phase 1
        print("📝 Creating Phase 1 model...")
        model = ConfigurableSectionLayoutGenerator(phase="phase1")
        
        original_info = model.get_model_info()
        print(f"   Original parameters: {original_info['total_parameters']:,}")
        
        # Upgrade to Phase 2
        print("🚀 Upgrading to Phase 2...")
        model.upgrade_to_phase("phase2", preserve_weights=True)
        
        upgraded_info = model.get_model_info()
        print(f"   New parameters: {upgraded_info['total_parameters']:,}")
        print(f"   Growth factor: {upgraded_info['total_parameters'] / original_info['total_parameters']:.1f}x")
        
        print("✅ Model upgrade successful!")
        return True
        
    except Exception as e:
        print(f"❌ Model upgrade failed: {e}")
        return False


def validate_configuration_consistency():
    """Validate that configurations are consistent and properly scaled."""
    
    print("\n🔍 VALIDATING CONFIGURATION CONSISTENCY")
    print("=" * 50)
    
    phases = ["phase1", "phase2", "phase3", "phase4"]
    
    # Check that parameters increase across phases
    param_counts = []
    for phase in phases:
        estimated_params = config_loader.estimate_parameters(phase=phase)
        param_counts.append(estimated_params)
        print(f"{phase}: ~{estimated_params/1e6:.1f}M parameters")
    
    # Validate increasing complexity
    for i in range(1, len(param_counts)):
        if param_counts[i] <= param_counts[i-1]:
            print(f"❌ Parameters not increasing: {phases[i-1]} → {phases[i]}")
            return False
    
    print("\n✅ Configuration scaling is consistent!")
    return True


def main():
    """Run all validation tests."""
    
    print("🧪 PHASE-BASED GROWTH STRATEGY VALIDATION")
    print("=" * 60)
    
    tests = [
        ("Phase Configurations", validate_phase_configs),
        ("Dataset Size Detection", validate_dataset_size_detection),
        ("Model Creation", validate_model_creation),
        ("Phase 1 Inference", test_phase1_inference),
        ("Model Upgrade", test_model_upgrade),
        ("Configuration Consistency", validate_configuration_consistency)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("🎯 VALIDATION SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<30} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("Your phase-based growth strategy is ready for implementation!")
        print("\nNext steps:")
        print("1. ✅ Start with Phase 1 for your 1,500 examples")
        print("2. ✅ Begin training with Phase 1 configuration")
        print("3. ✅ Scale to Phase 2 when you reach 2,500+ examples")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please review the errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 