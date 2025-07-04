#!/usr/bin/env python3
"""
Training Phase Detection Script

Analyzes your dataset size and recommends the optimal training phase
with cost estimates and training time predictions.

Usage:
    python3 examples/detect_training_phase.py --dataset_dir data/raw
    python3 examples/detect_training_phase.py --dataset_dir data/raw --output phase
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import config_loader


def count_dataset_samples(dataset_dir):
    """Count total samples in the dataset."""
    dataset_path = Path(dataset_dir)
    total_samples = 0
    splits_info = {}
    
    # Check for train/val/test structure
    for split in ['train', 'val', 'test']:
        split_dir = dataset_path / split
        if split_dir.exists() and split_dir.is_dir():
            # Count subdirectories (each subdirectory is an example)
            examples = [d for d in split_dir.iterdir() if d.is_dir()]
            splits_info[split] = len(examples)
            total_samples += len(examples)
    
    # If no splits found, count direct subdirectories
    if total_samples == 0:
        examples = [d for d in dataset_path.iterdir() if d.is_dir()]
        total_samples = len(examples)
        splits_info['unsplit'] = total_samples
    
    return total_samples, splits_info


def detect_optimal_phase(dataset_size):
    """Detect optimal training phase based on dataset size."""
    
    if dataset_size <= 2000:
        return "phase1"
    elif dataset_size <= 10000:
        return "phase2" 
    elif dataset_size <= 100000:
        return "phase3"
    else:
        return "phase4"


def get_phase_info(phase, dataset_size):
    """Get detailed information about a training phase."""
    
    phase_configs = {
        "phase1": {
            "name": "Micro-Scale Training",
            "dataset_range": "0-2K samples",
            "training_time": "2-4 hours",
            "cost_per_month": "$20-50",
            "gpu_requirements": "1x T4 (4GB)",
            "parameters": "1.2M",
            "expected_accuracy": "75-80%",
            "use_case": "Prototyping, proof-of-concept",
            "description": "Perfect for initial development and testing"
        },
        "phase2": {
            "name": "Small-Scale Training", 
            "dataset_range": "2.5K-10K samples",
            "training_time": "6-12 hours",
            "cost_per_month": "$100-200",
            "gpu_requirements": "1x V100 (16GB)",
            "parameters": "3.6M",
            "expected_accuracy": "82-87%",
            "use_case": "MVP, small production",
            "description": "Ideal for early production deployment"
        },
        "phase3": {
            "name": "Medium-Scale Training",
            "dataset_range": "25K-100K samples", 
            "training_time": "1-3 days",
            "cost_per_month": "$300-500",
            "gpu_requirements": "2-4x A100 (40GB)",
            "parameters": "12M",
            "expected_accuracy": "88-92%",
            "use_case": "Production service",
            "description": "Professional-grade model for production"
        },
        "phase4": {
            "name": "Large-Scale Training",
            "dataset_range": "100K+ samples",
            "training_time": "3-7 days", 
            "cost_per_month": "$800-1500",
            "gpu_requirements": "4-8x A100 (80GB)",
            "parameters": "50M+",
            "expected_accuracy": "92-96%",
            "use_case": "Enterprise deployment",
            "description": "Enterprise-level model with maximum capability"
        }
    }
    
    return phase_configs.get(phase, phase_configs["phase1"])


def print_phase_detection_results(dataset_size, splits_info, recommended_phase, output_mode=None):
    """Print comprehensive phase detection results."""
    
    if output_mode == "phase":
        # Simple output - just the phase name
        print(recommended_phase)
        return
    
    print("🔍 TRAINING PHASE DETECTION")
    print("=" * 60)
    
    # Dataset analysis
    print(f"📊 Dataset Analysis:")
    print(f"   Total samples: {dataset_size:,}")
    
    if 'unsplit' not in splits_info:
        print(f"   Dataset splits:")
        for split, count in splits_info.items():
            percentage = (count / dataset_size) * 100 if dataset_size > 0 else 0
            print(f"     • {split:5}: {count:,} samples ({percentage:.1f}%)")
    else:
        print(f"   Structure: Single directory (not split)")
    
    # Phase recommendation
    phase_info = get_phase_info(recommended_phase, dataset_size)
    
    print(f"\n🎯 Recommended Phase: {recommended_phase.upper()}")
    print(f"   Name: {phase_info['name']}")
    print(f"   Range: {phase_info['dataset_range']}")
    print(f"   Description: {phase_info['description']}")
    
    print(f"\n💰 Resource Requirements:")
    print(f"   Training time: {phase_info['training_time']}")
    print(f"   Estimated cost: {phase_info['cost_per_month']}/month")
    print(f"   GPU requirements: {phase_info['gpu_requirements']}")
    print(f"   Model parameters: {phase_info['parameters']}")
    
    print(f"\n📈 Expected Performance:")
    print(f"   Layout accuracy: {phase_info['expected_accuracy']}")
    print(f"   Use case: {phase_info['use_case']}")
    
    # Alternative phases
    print(f"\n🔄 Alternative Phases:")
    
    all_phases = ["phase1", "phase2", "phase3", "phase4"]
    for phase in all_phases:
        if phase != recommended_phase:
            alt_info = get_phase_info(phase, dataset_size)
            suitable = "✅" if is_phase_suitable(phase, dataset_size) else "❌"
            print(f"   {suitable} {phase.upper()}: {alt_info['name']} ({alt_info['dataset_range']})")
    
    # Training command
    print(f"\n🚀 Ready to Train:")
    print(f"   python3 scripts/train_model.py \\")
    print(f"     --dataset_dir data/raw \\") 
    print(f"     --output_dir models/experiment_1 \\")
    print(f"     --phase {recommended_phase}")
    
    print(f"\n   Or use auto-detection:")
    print(f"   python3 scripts/train_model.py \\")
    print(f"     --dataset_dir data/raw \\")
    print(f"     --output_dir models/experiment_1 \\")
    print(f"     --auto_phase")


def is_phase_suitable(phase, dataset_size):
    """Check if a phase is suitable for the given dataset size."""
    
    suitability = {
        "phase1": dataset_size <= 3000,
        "phase2": 1000 <= dataset_size <= 15000,
        "phase3": 5000 <= dataset_size <= 150000,
        "phase4": dataset_size >= 50000
    }
    
    return suitability.get(phase, False)


def print_scaling_advice(current_phase, dataset_size):
    """Print advice about scaling to larger phases."""
    
    print(f"\n💡 Scaling Advice:")
    
    if current_phase == "phase1":
        print(f"   • You're starting with Phase 1 - perfect for prototyping!")
        print(f"   • Consider upgrading to Phase 2 when you reach 2,500+ examples")
        print(f"   • Focus on data quality over quantity at this stage")
        print(f"   • Use aggressive data augmentation to maximize your small dataset")
        
    elif current_phase == "phase2":
        print(f"   • Phase 2 is great for MVP and small production deployments")
        print(f"   • Upgrade to Phase 3 when you reach 25,000+ examples")
        print(f"   • This phase offers good balance of performance and cost")
        
    elif current_phase == "phase3":
        print(f"   • Phase 3 provides production-ready performance")
        print(f"   • Consider Phase 4 only for enterprise-scale requirements (100K+ examples)")
        print(f"   • Excellent choice for most commercial applications")
        
    else:  # phase4
        print(f"   • Phase 4 represents state-of-the-art capability")
        print(f"   • You're at the cutting edge of model development")
        print(f"   • Focus on infrastructure and distributed training optimization")


def main():
    parser = argparse.ArgumentParser(description='Detect optimal training phase for your dataset')
    parser.add_argument('--dataset_dir', required=True, help='Path to dataset directory')
    parser.add_argument('--output', choices=['phase', 'full'], default='full',
                       help='Output format: "phase" for just phase name, "full" for detailed analysis')
    
    args = parser.parse_args()
    
    # Validate dataset directory
    if not os.path.exists(args.dataset_dir):
        print(f"❌ Error: Dataset directory not found: {args.dataset_dir}")
        sys.exit(1)
    
    try:
        # Count dataset samples
        dataset_size, splits_info = count_dataset_samples(args.dataset_dir)
        
        if dataset_size == 0:
            print(f"❌ Error: No examples found in {args.dataset_dir}")
            print(f"💡 Make sure your dataset has the correct structure:")
            print(f"   data/raw/train/example_001/")
            print(f"   data/raw/val/example_002/")
            print(f"   data/raw/test/example_003/")
            sys.exit(1)
        
        # Detect optimal phase
        recommended_phase = detect_optimal_phase(dataset_size)
        
        # Print results
        print_phase_detection_results(dataset_size, splits_info, recommended_phase, args.output)
        
        # Print scaling advice (only in full mode)
        if args.output == 'full':
            print_scaling_advice(recommended_phase, dataset_size)
            
            print("\n" + "=" * 60)
            print("✅ Phase detection completed!")
            print("🚀 Your dataset is ready for training!")
            print("=" * 60)
    
    except Exception as e:
        print(f"❌ Error during phase detection: {e}")
        if args.output == 'full':
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()