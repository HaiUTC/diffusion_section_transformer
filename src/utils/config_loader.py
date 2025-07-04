"""
Configuration Loader for Phase-Based Model Development
Provides dynamic model scaling based on dataset size and training phase.
"""

import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    """Model configuration dataclass for type safety."""
    d_model: int
    n_heads: int
    n_layers: int
    dropout: float
    vision: Dict[str, Any]
    structure: Dict[str, Any]
    layout: Dict[str, Any]
    diffusion: Dict[str, Any]


@dataclass
class TrainingConfig:
    """Training configuration dataclass."""
    batch_size: int
    learning_rate: float
    weight_decay: float
    epochs: int
    warmup_steps: int
    dropout: float
    label_smoothing: float
    gradient_clip: float
    augmentation: Dict[str, Any]


@dataclass
class ValidationConfig:
    """Validation configuration dataclass."""
    split: float
    patience: int
    min_delta: float


class PhaseConfigLoader:
    """Loads and manages phase-specific configurations."""
    
    def __init__(self, config_dir: str = None):
        if config_dir is None:
            # Find project root by looking for specific files that exist in the project root
            current_path = Path(__file__).resolve()
            project_root = current_path
            
            # Walk up the directory tree to find the project root
            while project_root.parent != project_root:
                if (project_root / "configs").exists() and (project_root / "src").exists():
                    break
                project_root = project_root.parent
            
            self.config_dir = project_root / "configs"
        else:
            self.config_dir = Path(config_dir)
        
        self.phase_configs = {}
        self._load_all_phases()
    
    def _load_all_phases(self):
        """Load all phase configurations."""
        phases = ["phase1", "phase2", "phase3", "phase4"]
        
        for phase in phases:
            config_path = self.config_dir / f"{phase}_config.yaml"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.phase_configs[phase] = yaml.safe_load(f)
                print(f"🔧 Loaded {phase} config from {config_path}")
            else:
                print(f"⚠️  Warning: {config_path} not found")
    
    def get_phase_by_dataset_size(self, dataset_size: int) -> str:
        """Determine phase based on dataset size."""
        if dataset_size <= 2000:
            return "phase1"
        elif dataset_size < 5000:
            return "phase2"
        elif dataset_size <= 10000:
            return "phase3"
        else:
            return "phase4"
    
    def load_config(self, phase: Optional[str] = None, dataset_size: Optional[int] = None) -> Dict[str, Any]:
        """Load configuration for specific phase or dataset size."""
        if phase is None and dataset_size is not None:
            phase = self.get_phase_by_dataset_size(dataset_size)
        elif phase is None:
            phase = "phase1"  # Default to smallest model
        
        if phase not in self.phase_configs:
            raise ValueError(f"Phase {phase} configuration not found")
        
        return self.phase_configs[phase]
    
    def get_model_config(self, phase: Optional[str] = None, dataset_size: Optional[int] = None) -> ModelConfig:
        """Get model configuration as typed dataclass."""
        config = self.load_config(phase, dataset_size)
        model_cfg = config["model"]
        
        return ModelConfig(
            d_model=model_cfg["d_model"],
            n_heads=model_cfg["n_heads"],
            n_layers=model_cfg["n_layers"],
            dropout=model_cfg["dropout"],
            vision=model_cfg["vision"],
            structure=model_cfg["structure"],
            layout=model_cfg["layout"],
            diffusion=model_cfg["diffusion"]
        )
    
    def get_training_config(self, phase: Optional[str] = None, dataset_size: Optional[int] = None) -> TrainingConfig:
        """Get training configuration as typed dataclass."""
        config = self.load_config(phase, dataset_size)
        train_cfg = config["training"]
        
        return TrainingConfig(
            batch_size=train_cfg["batch_size"],
            learning_rate=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
            epochs=train_cfg["epochs"],
            warmup_steps=train_cfg["warmup_steps"],
            dropout=train_cfg["dropout"],
            label_smoothing=train_cfg["label_smoothing"],
            gradient_clip=train_cfg["gradient_clip"],
            augmentation=train_cfg["augmentation"]
        )
    
    def get_validation_config(self, phase: Optional[str] = None, dataset_size: Optional[int] = None) -> ValidationConfig:
        """Get validation configuration as typed dataclass."""
        config = self.load_config(phase, dataset_size)
        val_cfg = config["validation"]
        
        return ValidationConfig(
            split=val_cfg["split"],
            patience=val_cfg["patience"],
            min_delta=val_cfg["min_delta"]
        )
    
    def estimate_parameters(self, phase: Optional[str] = None, dataset_size: Optional[int] = None) -> int:
        """Estimate model parameters for given configuration."""
        model_cfg = self.get_model_config(phase, dataset_size)
        
        # Rough parameter estimation (simplified)
        d_model = model_cfg.d_model
        n_layers = model_cfg.n_layers
        n_heads = model_cfg.n_heads
        vocab_size = model_cfg.structure["vocab_size"]
        
        # Transformer parameters estimation
        attention_params = n_layers * (4 * d_model**2 + 4 * d_model)  # Q,K,V,O projections
        ffn_params = n_layers * (8 * d_model**2 + 8 * d_model)  # FFN layers
        embedding_params = vocab_size * d_model + 1000 * d_model  # Token + pos embeddings
        
        total_params = attention_params + ffn_params + embedding_params
        return int(total_params)
    
    def print_phase_summary(self, phase: Optional[str] = None, dataset_size: Optional[int] = None):
        """Print summary of phase configuration."""
        if phase is None and dataset_size is not None:
            phase = self.get_phase_by_dataset_size(dataset_size)
        
        model_cfg = self.get_model_config(phase, dataset_size)
        train_cfg = self.get_training_config(phase, dataset_size)
        estimated_params = self.estimate_parameters(phase, dataset_size)
        
        print(f"\n=== {phase.upper()} CONFIGURATION ===")
        print(f"Model Parameters: ~{estimated_params/1e6:.1f}M")
        print(f"Dimensions: {model_cfg.d_model} | Heads: {model_cfg.n_heads} | Layers: {model_cfg.n_layers}")
        print(f"Batch Size: {train_cfg.batch_size} | Learning Rate: {train_cfg.learning_rate}")
        print(f"Max Sequence Length: {model_cfg.structure['max_length']}")
        print(f"Vision Patch Size: {model_cfg.vision['patch_size']}")
        print(f"Training Epochs: {train_cfg.epochs}")
        print("=" * 40)


# Global config loader instance
config_loader = PhaseConfigLoader()


def get_config_for_dataset_size(dataset_size: int) -> Dict[str, Any]:
    """Convenience function to get config by dataset size."""
    return config_loader.load_config(dataset_size=dataset_size)


def get_current_phase_config() -> Dict[str, Any]:
    """Get Phase 1 configuration (current user phase)."""
    return config_loader.load_config(phase="phase1") 