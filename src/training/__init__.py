"""
Training Strategies & Loss Functions - Step 5 Implementation

This package implements comprehensive training strategies tailored to each dataset size phase,
leveraging cutting-edge research in few-shot learning, curriculum learning, and multimodal optimization.

Implemented Components:
1. Phase-Specific Training Strategies (Phase 1-4)
2. Data Augmentation Pipelines
3. Loss Functions & Scheduling
4. Curriculum Learning
5. Few-Shot Learning Integration
6. Training Optimization Techniques

Reference: Step 5 specifications from instruction.md
"""

# Phase-specific training strategies
from .phase_strategies import (
    PhaseTrainingStrategy, Phase1MicroScaleStrategy, Phase2SmallScaleStrategy,
    Phase3MediumScaleStrategy, Phase4LargeScaleStrategy, create_phase_strategy
)

# Data augmentation pipelines
from .data_augmentation import (
    AggressiveAugmentationConfig, ScreenshotAugmentationPipeline, StructureAugmentationPipeline,
    CombinedAugmentationPipeline, create_augmentation_config, demonstrate_augmentation_pipeline
)

# Loss functions and scheduling
from .loss_functions import (
    ElementCombinationLoss, VarianceAwareLossScheduler, ModalityAwareLossWeighting,
    MultiScaleConsistencyLoss, MultiTaskLossFunction, create_phase_loss_function
)

# Utility functions for demo and testing
def get_phase_summary(strategy: PhaseTrainingStrategy) -> dict:
    """Get summary information about a training strategy."""
    return {
        'phase_name': strategy.__class__.__name__,
        'dataset_size_range': strategy.get_dataset_size_range(),
        'training_techniques': strategy.get_training_techniques(),
        'configuration': strategy.get_configuration_summary(),
        'special_features': strategy.get_special_features()
    }

# Export all public components
__all__ = [
    # Phase strategies
    'PhaseTrainingStrategy',
    'Phase1MicroScaleStrategy', 
    'Phase2SmallScaleStrategy',
    'Phase3MediumScaleStrategy',
    'Phase4LargeScaleStrategy',
    'create_phase_strategy',
    'get_phase_summary',
    
    # Data augmentation
    'AggressiveAugmentationConfig',
    'ScreenshotAugmentationPipeline',
    'StructureAugmentationPipeline', 
    'CombinedAugmentationPipeline',
    'create_augmentation_config',
    'demonstrate_augmentation_pipeline',
    
    # Loss functions
    'ElementCombinationLoss',
    'VarianceAwareLossScheduler',
    'ModalityAwareLossWeighting',
    'MultiScaleConsistencyLoss',
    'MultiTaskLossFunction',
    'create_phase_loss_function',
] 