"""
Phase-Specific Training Strategies - Step 5 Implementation

This module implements comprehensive training strategies tailored to each dataset size phase:
- Phase 1: Micro-Scale Training (0-2,000 samples) - Aggressive augmentation & few-shot learning
- Phase 2: Small-Scale Training (2,500-10,000 samples) - Curriculum learning & TDC training  
- Phase 3: Medium-Scale Training (25,000-100,000 samples) - Standard diffusion training
- Phase 4: Large-Scale Training (100,000+ samples) - Scalable infrastructure & production techniques

Reference: Step 5 specifications from instruction.md
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from typing import Dict, List, Optional, Any, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
import math


@dataclass
class PhaseConfig:
    """Configuration for phase-specific training."""
    phase_name: str
    dataset_size_range: Tuple[int, int]
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    dropout_rate: float
    patience: int
    min_improvement: float
    validation_split: float
    augmentation_factor: int
    enable_few_shot: bool = False
    enable_curriculum: bool = False
    enable_mixed_precision: bool = False
    enable_distributed: bool = False


class PhaseTrainingStrategy(ABC):
    """
    Abstract base class for phase-specific training strategies.
    Each phase implements different optimization approaches based on data availability.
    """
    
    def __init__(self, config: PhaseConfig):
        self.config = config
        self.current_epoch = 0
        self.best_validation_loss = float('inf')
        self.patience_counter = 0
        
    @abstractmethod
    def configure_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Configure optimizer for this phase."""
        pass
    
    @abstractmethod
    def configure_scheduler(self, optimizer: optim.Optimizer) -> Any:
        """Configure learning rate scheduler for this phase."""
        pass
    
    @abstractmethod
    def configure_loss_function(self) -> nn.Module:
        """Configure loss function for this phase."""
        pass
    
    @abstractmethod
    def should_stop_early(self, validation_loss: float) -> bool:
        """Determine if training should stop early."""
        pass
    
    @abstractmethod
    def get_training_techniques(self) -> List[str]:
        """Get list of training techniques used in this phase."""
        pass

    def get_dataset_size_range(self) -> Tuple[int, int]:
        """Get the dataset size range for this phase."""
        return self.config.dataset_size_range
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get a summary of key configuration parameters."""
        return {
            'epochs': self.config.epochs,
            'batch_size': self.config.batch_size,
            'learning_rate': self.config.learning_rate,
            'weight_decay': self.config.weight_decay,
            'dropout_rate': self.config.dropout_rate,
            'augmentation_factor': self.config.augmentation_factor,
            'patience': self.config.patience,
            'validation_split': self.config.validation_split
        }
    
    def get_special_features(self) -> Dict[str, bool]:
        """Get special features enabled for this phase."""
        return {
            'few_shot_learning': self.config.enable_few_shot,
            'curriculum_learning': self.config.enable_curriculum,
            'mixed_precision': self.config.enable_mixed_precision,
            'distributed_training': self.config.enable_distributed,
            'variance_aware_loss': hasattr(self, 'enable_variance_aware') and self.enable_variance_aware,
            'progressive_training': hasattr(self, 'enable_progressive') and self.enable_progressive
        }


class Phase1MicroScaleStrategy(PhaseTrainingStrategy):
    """
    Phase 1: Micro-Scale Training (0-2,000 samples)
    
    Techniques:
    - Aggressive data augmentation (50x augmentation)
    - Few-shot diffusion models (FSDM) integration
    - Transfer learning with pre-trained components
    - Variance-aware loss scheduling
    - K-fold cross-validation
    - High regularization (dropout 0.3-0.5, L2 1e-4 to 1e-5)
    """
    
    def __init__(self, config: PhaseConfig):
        super().__init__(config)
        self.augmentation_factor = 50  # 50x augmentation for 2k → 100k samples
        self.regularization_strength = 1e-4
        self.variance_tracking_window = 100
        self.loss_variance_history = []
        
    def configure_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Configure optimizer with high regularization for small data."""
        return optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.regularization_strength,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    def configure_scheduler(self, optimizer: optim.Optimizer) -> Any:
        """Linear warmup followed by cosine annealing for stable training."""
        warmup_steps = self.config.epochs // 10  # 10% warmup
        
        warmup_scheduler = LinearLR(
            optimizer, 
            start_factor=0.1, 
            end_factor=1.0, 
            total_iters=warmup_steps
        )
        
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config.epochs - warmup_steps,
            eta_min=self.config.learning_rate * 0.01
        )
        
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
    
    def configure_loss_function(self) -> nn.Module:
        """Variance-aware loss with high regularization."""
        from .loss_functions import VarianceAwareLossScheduler, ElementCombinationLoss
        
        base_loss = ElementCombinationLoss(
            element_vocab_size=200,
            regularization_weight=self.regularization_strength
        )
        
        return VarianceAwareLossScheduler(
            base_loss=base_loss,
            variance_window=self.variance_tracking_window,
            adaptive_weighting=True
        )
    
    def should_stop_early(self, validation_loss: float) -> bool:
        """Early stopping with high patience for small data."""
        improved = validation_loss < (self.best_validation_loss - self.config.min_improvement)
        
        if improved:
            self.best_validation_loss = validation_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config.patience
    
    def get_training_techniques(self) -> List[str]:
        """List of techniques used in Phase 1."""
        return [
            "Aggressive Data Augmentation (50x)",
            "Few-Shot Diffusion Models (FSDM)",
            "Transfer Learning (ViT-B/16, BERT-base)",
            "Variance-Aware Loss Scheduling",
            "K-Fold Cross-Validation (5-fold)",
            "High Regularization (L2=1e-4, Dropout=0.3-0.5)",
            "Progressive Unfreezing",
            "Linear Warmup + Cosine Annealing"
        ]
    
    def get_augmentation_config(self) -> Dict[str, Any]:
        """Get augmentation configuration for Phase 1."""
        return {
            "screenshot_augmentation": {
                "rotation_range": (-15, 15),
                "scale_range": (0.8, 1.2),
                "translation_range": (-0.1, 0.1),
                "brightness_range": (0.7, 1.3),
                "contrast_range": (0.8, 1.2),
                "saturation_range": (0.9, 1.1),
                "resolution_scales": [256, 384, 512, 768, 1024]
            },
            "structure_augmentation": {
                "enable_reordering": True,
                "class_substitution_prob": 0.3,
                "hierarchy_modification_prob": 0.2,
                "content_abstraction_prob": 0.4
            },
            "augmentation_factor": self.augmentation_factor
        }


class Phase2SmallScaleStrategy(PhaseTrainingStrategy):
    """
    Phase 2: Small-Scale Training (2,500-10,000 samples)
    
    Techniques:
    - Curriculum learning with progressive difficulty
    - Two-Stage Divide-and-Conquer (TDC) training
    - Progressive data dropout
    - Modality-aware loss weighting
    - Multi-scale consistency loss
    """
    
    def __init__(self, config: PhaseConfig):
        super().__init__(config)
        self.curriculum_stages = 3
        self.current_stage = 1
        self.stage_epochs = [20, 30, 50]  # Epochs per curriculum stage
        self.stage_learning_rates = [1e-3, 5e-4, 1e-4]
        
    def configure_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Stage-specific optimizer configuration."""
        current_lr = self.stage_learning_rates[self.current_stage - 1]
        
        return optim.AdamW(
            model.parameters(),
            lr=current_lr,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999)
        )
    
    def configure_scheduler(self, optimizer: optim.Optimizer) -> Any:
        """Stage-specific learning rate scheduling."""
        stage_epochs = self.stage_epochs[self.current_stage - 1]
        
        return CosineAnnealingLR(
            optimizer,
            T_max=stage_epochs,
            eta_min=self.stage_learning_rates[self.current_stage - 1] * 0.1
        )
    
    def configure_loss_function(self) -> nn.Module:
        """Multi-task loss with modality-aware weighting."""
        from .loss_functions import (
            MultiTaskLossFunction, ModalityAwareLossWeighting, 
            MultiScaleConsistencyLoss
        )
        
        # Stage-specific loss configuration
        if self.current_stage == 1:
            # Simple layouts: focus on structural accuracy
            return MultiTaskLossFunction(
                enable_aesthetic_loss=False,
                enable_alignment_loss=True,
                enable_diversity_loss=False,
                enable_props_loss=False
            )
        elif self.current_stage == 2:
            # Medium complexity: add aesthetic constraints
            return MultiTaskLossFunction(
                enable_aesthetic_loss=True,
                enable_alignment_loss=True,
                enable_diversity_loss=False,
                enable_props_loss=True
            )
        else:
            # Complex layouts: full loss function
            modality_aware_loss = ModalityAwareLossWeighting(
                visual_weight=0.4,
                structural_weight=0.4,
                geometric_weight=0.2
            )
            
            multiscale_loss = MultiScaleConsistencyLoss(
                scales=[256, 512, 768]
            )
            
            return MultiTaskLossFunction(
                enable_aesthetic_loss=True,
                enable_alignment_loss=True,
                enable_diversity_loss=True,
                enable_props_loss=True,
                modality_weighter=modality_aware_loss,
                consistency_loss=multiscale_loss
            )
    
    def should_stop_early(self, validation_loss: float) -> bool:
        """Stage-aware early stopping."""
        improved = validation_loss < (self.best_validation_loss - self.config.min_improvement)
        
        if improved:
            self.best_validation_loss = validation_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            # Stage-specific patience
            stage_patience = [15, 12, 10][self.current_stage - 1]
            return self.patience_counter >= stage_patience
    
    def advance_curriculum_stage(self) -> bool:
        """Advance to next curriculum stage if ready."""
        stage_epochs = self.stage_epochs[self.current_stage - 1]
        
        if self.current_epoch >= stage_epochs and self.current_stage < self.curriculum_stages:
            self.current_stage += 1
            self.current_epoch = 0
            self.patience_counter = 0
            return True
        return False
    
    def get_training_techniques(self) -> List[str]:
        """List of techniques used in Phase 2."""
        return [
            "Curriculum Learning (3-stage progressive difficulty)",
            "Two-Stage Divide-and-Conquer (TDC) Training",
            "Progressive Data Dropout",
            "Modality-Aware Loss Weighting",
            "Multi-Scale Consistency Loss",
            "Stage-Specific Learning Rates",
            "Adaptive Early Stopping"
        ]
    
    def get_curriculum_config(self) -> Dict[str, Any]:
        """Get curriculum learning configuration."""
        return {
            "stage_1": {
                "complexity": "simple",
                "max_elements": 5,
                "learning_rate": 1e-3,
                "epochs": 20,
                "focus": "basic element mapping and positioning"
            },
            "stage_2": {
                "complexity": "medium", 
                "max_elements": 15,
                "learning_rate": 5e-4,
                "epochs": 30,
                "focus": "compound element combinations"
            },
            "stage_3": {
                "complexity": "complex",
                "max_elements": 32,
                "learning_rate": 1e-4,
                "epochs": 50,
                "focus": "advanced background property handling"
            }
        }


class Phase3MediumScaleStrategy(PhaseTrainingStrategy):
    """
    Phase 3: Medium-Scale Training (25,000-100,000 samples)
    
    Techniques:
    - Standard diffusion training with CFG
    - Mixed-precision training (FP16)
    - Advanced regularization (stochastic depth, noise injection)
    - Label smoothing
    - Larger batch sizes
    """
    
    def __init__(self, config: PhaseConfig):
        super().__init__(config)
        self.guidance_scale = 7.5
        self.enable_mixed_precision = True
        self.stochastic_depth_prob = 0.1
        self.noise_injection_std = 0.01
        
    def configure_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Standard AdamW with mixed precision support."""
        return optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    def configure_scheduler(self, optimizer: optim.Optimizer) -> Any:
        """Cosine annealing with warmup."""
        warmup_steps = 10000
        total_steps = self.config.epochs * 1000  # Approximate steps per epoch
        
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=self.config.learning_rate * 0.01
        )
        
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
    
    def configure_loss_function(self) -> nn.Module:
        """Standard multi-task loss with label smoothing."""
        from .loss_functions import MultiTaskLossFunction
        
        return MultiTaskLossFunction(
            enable_aesthetic_loss=True,
            enable_alignment_loss=True,
            enable_diversity_loss=True,
            enable_props_loss=True,
            label_smoothing=0.1,
            guidance_scale=self.guidance_scale
        )
    
    def should_stop_early(self, validation_loss: float) -> bool:
        """Standard early stopping with moderate patience."""
        improved = validation_loss < (self.best_validation_loss - self.config.min_improvement)
        
        if improved:
            self.best_validation_loss = validation_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config.patience
    
    def get_training_techniques(self) -> List[str]:
        """List of techniques used in Phase 3."""
        return [
            "Standard Diffusion Training",
            "Classifier-Free Guidance (CFG=7.5)",
            "Mixed-Precision Training (FP16)",
            "Stochastic Depth Regularization",
            "Noise Injection",
            "Label Smoothing (0.1)",
            "Large Batch Training (128-256)",
            "Cosine Annealing LR Schedule"
        ]


class Phase4LargeScaleStrategy(PhaseTrainingStrategy):
    """
    Phase 4: Large-Scale Training (100,000+ samples)
    
    Techniques:
    - Multi-GPU distributed training
    - Gradient accumulation for large effective batch sizes
    - Exponential moving average (EMA) of model weights
    - Advanced production-ready techniques
    - Comprehensive validation strategies
    """
    
    def __init__(self, config: PhaseConfig):
        super().__init__(config)
        self.enable_distributed = True
        self.effective_batch_size = 1024  # Via gradient accumulation
        self.gradient_accumulation_steps = 4
        self.ema_decay = 0.9999
        self.gradient_clip_norm = 1.0
        
    def configure_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Scaled optimizer for large-scale training."""
        # Linear learning rate scaling based on effective batch size
        scaled_lr = self.config.learning_rate * (self.effective_batch_size / 256)
        
        return optim.AdamW(
            model.parameters(),
            lr=scaled_lr,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    def configure_scheduler(self, optimizer: optim.Optimizer) -> Any:
        """Production-grade learning rate scheduling."""
        warmup_steps = 10000
        total_steps = self.config.epochs * 2000  # Higher steps for large scale
        
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=self.config.learning_rate * 0.001
        )
        
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
    
    def configure_loss_function(self) -> nn.Module:
        """Production-ready comprehensive multi-task loss."""
        from .loss_functions import MultiTaskLossFunction, ModalityAwareLossWeighting
        
        modality_weighter = ModalityAwareLossWeighting(
            visual_weight=0.35,
            structural_weight=0.35,
            geometric_weight=0.2,
            aesthetic_weight=0.1,
            uncertainty_based=True
        )
        
        return MultiTaskLossFunction(
            enable_aesthetic_loss=True,
            enable_alignment_loss=True,
            enable_diversity_loss=True,
            enable_props_loss=True,
            modality_weighter=modality_weighter,
            dynamic_weighting=True,
            gradient_clipping=self.gradient_clip_norm
        )
    
    def should_stop_early(self, validation_loss: float) -> bool:
        """Conservative early stopping for production training."""
        improved = validation_loss < (self.best_validation_loss - self.config.min_improvement)
        
        if improved:
            self.best_validation_loss = validation_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            # Higher patience for large-scale training
            return self.patience_counter >= (self.config.patience * 2)
    
    def get_training_techniques(self) -> List[str]:
        """List of techniques used in Phase 4."""
        return [
            "Multi-GPU Distributed Training (DDP)",
            "Gradient Accumulation (Effective BS=1024)",
            "Linear Learning Rate Scaling",
            "Exponential Moving Average (EMA)",
            "Gradient Norm Clipping (1.0)",
            "Dynamic Loss Weighting",
            "Uncertainty-Based Modality Weighting",
            "Production Validation Strategy (85/10/5 split)"
        ]
    
    def get_distributed_config(self) -> Dict[str, Any]:
        """Get distributed training configuration."""
        return {
            "backend": "nccl",
            "world_size": 4,  # 4-8 GPUs
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_batch_size": self.effective_batch_size,
            "sync_batch_norm": True,
            "find_unused_parameters": False
        }


def create_phase_strategy(phase: str, dataset_size: int) -> PhaseTrainingStrategy:
    """
    Factory function to create appropriate training strategy based on phase and dataset size.
    
    Args:
        phase: Phase name ("phase1", "phase2", "phase3", "phase4")
        dataset_size: Size of training dataset
        
    Returns:
        Appropriate PhaseTrainingStrategy instance
    """
    # Phase configurations based on instruction.md specifications
    configs = {
        "phase1": PhaseConfig(
            phase_name="phase1",
            dataset_size_range=(0, 2000),
            epochs=100,
            batch_size=8,
            learning_rate=1e-4,
            weight_decay=1e-4,
            dropout_rate=0.4,
            patience=15,
            min_improvement=0.001,
            validation_split=0.2,  # 5-fold CV
            augmentation_factor=50,
            enable_few_shot=True
        ),
        "phase2": PhaseConfig(
            phase_name="phase2", 
            dataset_size_range=(2500, 10000),
            epochs=100,
            batch_size=16,
            learning_rate=5e-4,
            weight_decay=1e-3,
            dropout_rate=0.3,
            patience=12,
            min_improvement=0.001,
            validation_split=0.1,  # 80/10/10 split
            augmentation_factor=10,
            enable_curriculum=True
        ),
        "phase3": PhaseConfig(
            phase_name="phase3",
            dataset_size_range=(25000, 100000),
            epochs=50,
            batch_size=128,
            learning_rate=1e-4,
            weight_decay=1e-2,
            dropout_rate=0.2,
            patience=8,
            min_improvement=0.0005,
            validation_split=0.15,  # 80/15/5 split
            augmentation_factor=5,
            enable_mixed_precision=True
        ),
        "phase4": PhaseConfig(
            phase_name="phase4",
            dataset_size_range=(100000, float('inf')),
            epochs=30,
            batch_size=256,
            learning_rate=1e-4,
            weight_decay=1e-2,
            dropout_rate=0.1,
            patience=5,
            min_improvement=0.0001,
            validation_split=0.1,  # 85/10/5 split  
            augmentation_factor=2,
            enable_distributed=True
        )
    }
    
    # Auto-detect phase if not specified
    if phase == "auto":
        if dataset_size <= 2000:
            phase = "phase1"
        elif dataset_size <= 10000:
            phase = "phase2"
        elif dataset_size <= 100000:
            phase = "phase3"
        else:
            phase = "phase4"
    
    config = configs[phase]
    
    # Create appropriate strategy
    if phase == "phase1":
        return Phase1MicroScaleStrategy(config)
    elif phase == "phase2":
        return Phase2SmallScaleStrategy(config)
    elif phase == "phase3":
        return Phase3MediumScaleStrategy(config)
    elif phase == "phase4":
        return Phase4LargeScaleStrategy(config)
    else:
        raise ValueError(f"Unknown phase: {phase}")


def get_phase_summary(strategy: PhaseTrainingStrategy) -> Dict[str, Any]:
    """Get comprehensive summary of training strategy."""
    return {
        "phase_name": strategy.config.phase_name,
        "dataset_size_range": strategy.get_dataset_size_range(),
        "training_techniques": strategy.get_training_techniques(),
        "configuration": strategy.get_configuration_summary(),
        "special_features": strategy.get_special_features()
    } 