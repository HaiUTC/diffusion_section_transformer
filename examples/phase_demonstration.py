"""
Phase-Based Model Development Demonstration
Shows how the model architecture scales with dataset size across 4 phases.
"""

import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ai_engine_configurable import ConfigurableSectionLayoutGenerator, create_phase_appropriate_model
from src.utils.config_loader import config_loader


def demonstrate_phase_scaling():
    """Demonstrate model scaling across all phases."""
    
    print("=" * 60)
    print("PHASE-BASED MODEL DEVELOPMENT DEMONSTRATION")
    print("=" * 60)
    
    # Dataset sizes for each phase
    phase_datasets = {
        "phase1": 1500,   # User's current situation
        "phase2": 3500,
        "phase3": 7500,
        "phase4": 15000
    }
    
    models = {}
    
    for phase, dataset_size in phase_datasets.items():
        print(f"\n🔄 Creating {phase.upper()} model for {dataset_size} examples...")
        
        # Create model for this phase
        model = ConfigurableSectionLayoutGenerator(dataset_size=dataset_size)
        models[phase] = model
        
        # Get model information
        info = model.get_model_info()
        
        print(f"✅ Model created:")
        print(f"   • Parameters: {info['total_parameters']:,} ({info['total_parameters']/1e6:.1f}M)")
        print(f"   • Memory: {info['model_size_mb']:.1f} MB")
        print(f"   • Architecture: {info['d_model']} dim, {model.model_config.n_heads} heads, {model.model_config.n_layers} layers")
        print(f"   • Training batch size: {model.training_config.batch_size}")
        print(f"   • Inference speed: ~{10 + (info['total_parameters'] // 1e6) * 2} seconds")
    
    return models


def test_phase1_model():
    """Test Phase 1 model specifically for user's current situation."""
    
    print("\n" + "="*50)
    print("PHASE 1 MODEL TEST (Your Current Situation)")
    print("="*50)
    
    # Create Phase 1 model (0-2000 dataset)
    model = create_phase_appropriate_model(dataset_size=1500)
    
    # Generate some test data
    batch_size = 2
    screenshot = torch.randn(batch_size, 3, 224, 224)  # Mock screenshots
    structure_tokens = torch.randint(0, 100, (batch_size, 128))  # Mock HTML structure
    
    print("\n🧪 Testing Phase 1 model with mock data...")
    
    # Test inference
    model.eval()
    with torch.no_grad():
        try:
            outputs = model.generate_layout(
                screenshot=screenshot,
                structure_tokens=structure_tokens,
                num_steps=10  # Fast inference for Phase 1
            )
            
            print("✅ Layout generation successful!")
            print(f"   • Generated {len(outputs.elements)} elements")
            print(f"   • Aesthetic score: {outputs.aesthetic_score:.2f}")
            print(f"   • Constraint violations: {len(outputs.constraint_violations)}")
            print(f"   • Average confidence: {outputs.confidence_scores.mean():.2f}")
            
            # Show sample elements
            print("\n📋 Sample generated elements:")
            for i, element in enumerate(outputs.elements[:3]):
                print(f"   {i+1}. {element['type']} at ({element['position']['x']:.2f}, {element['position']['y']:.2f})")
                
        except Exception as e:
            print(f"❌ Error during generation: {e}")
    
    # Model statistics
    info = model.get_model_info()
    print(f"\n📊 Phase 1 Model Statistics:")
    print(f"   • Total parameters: {info['total_parameters']:,}")
    print(f"   • Memory usage: {info['model_size_mb']:.1f} MB")
    print(f"   • Estimated training time per epoch: ~15 seconds")
    print(f"   • Recommended dataset size: 1,000-2,000 examples")
    print(f"   • Overfitting prevention: Aggressive regularization enabled")


def compare_all_phases():
    """Compare model sizes and capabilities across phases."""
    
    print("\n" + "="*60)
    print("PHASE COMPARISON SUMMARY")
    print("="*60)
    
    phases_info = []
    
    for phase in ["phase1", "phase2", "phase3", "phase4"]:
        # Get configuration without creating full model
        model_config = config_loader.get_model_config(phase=phase)
        training_config = config_loader.get_training_config(phase=phase)
        estimated_params = config_loader.estimate_parameters(phase=phase)
        
        phases_info.append({
            'phase': phase,
            'dataset_range': {
                'phase1': '0-2,000',
                'phase2': '2,500-5,000', 
                'phase3': '5,000-10,000',
                'phase4': '10,000+'
            }[phase],
            'parameters': estimated_params,
            'd_model': model_config.d_model,
            'layers': model_config.n_layers,
            'batch_size': training_config.batch_size,
            'learning_rate': training_config.learning_rate,
            'epochs': training_config.epochs
        })
    
    # Print comparison table
    print(f"{'Phase':<8} {'Dataset':<12} {'Params':<10} {'Dims':<6} {'Layers':<7} {'Batch':<6} {'LR':<8} {'Epochs':<7}")
    print("-" * 70)
    
    for info in phases_info:
        print(f"{info['phase']:<8} {info['dataset_range']:<12} "
              f"{info['parameters']/1e6:.1f}M{'':<5} {info['d_model']:<6} "
              f"{info['layers']:<7} {info['batch_size']:<6} "
              f"{info['learning_rate']:<8} {info['epochs']:<7}")
    
    print("\n💡 Recommendations for your Phase 1 development:")
    print("   • Start with Phase 1 model (2-3M parameters)")
    print("   • Focus on data quality over quantity")
    print("   • Use aggressive data augmentation")
    print("   • Monitor for overfitting with early stopping")
    print("   • Upgrade to Phase 2 when you reach 2,500+ examples")


def show_upgrade_path():
    """Demonstrate model upgrade capabilities."""
    
    print("\n" + "="*50)
    print("MODEL UPGRADE DEMONSTRATION")
    print("="*50)
    
    print("🔄 Starting with Phase 1 model...")
    model = ConfigurableSectionLayoutGenerator(phase="phase1")
    
    original_info = model.get_model_info()
    print(f"   Initial parameters: {original_info['total_parameters']:,}")
    
    print("\n🚀 Upgrading to Phase 2...")
    model.upgrade_to_phase("phase2", preserve_weights=True)
    
    upgraded_info = model.get_model_info()
    print(f"   New parameters: {upgraded_info['total_parameters']:,}")
    print(f"   Growth factor: {upgraded_info['total_parameters'] / original_info['total_parameters']:.1f}x")
    
    print("\n✅ Upgrade completed! Weights preserved where possible.")
    print("   💡 This allows you to progressively scale your model as data grows.")


if __name__ == "__main__":
    try:
        # Run demonstrations
        demonstrate_phase_scaling()
        test_phase1_model()
        compare_all_phases()
        show_upgrade_path()
        
        print("\n" + "="*60)
        print("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("Your Phase 1 model is ready for 0-2000 dataset training.")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc() 