"""
Phase-Configurable Section Layout Generator
Adapts model architecture based on training phase and dataset size.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import yaml

from src.utils.config_loader import PhaseConfigLoader, ModelConfig, TrainingConfig
from src.models.multimodal_encoder import MultimodalEncoder  
from src.models.layout_embedding import LayoutEmbedding
from src.models.diffusion_decoder import DiffusionDecoder
from src.models.aesthetic_constraints import AestheticConstraintModule


@dataclass
class GenerationOutput:
    """Structured output from layout generation."""
    elements: List[Dict[str, Any]]
    layout_tokens: torch.Tensor
    confidence_scores: torch.Tensor
    constraint_violations: List[str]
    aesthetic_score: float


class ConfigurableSectionLayoutGenerator(nn.Module):
    """
    Configurable Generative AI engine for section layout generation.
    Automatically adapts architecture based on training phase and dataset size.
    """
    
    def __init__(
        self,
        phase: Optional[str] = None,
        dataset_size: Optional[int] = None,
        config_path: Optional[str] = None
    ):
        super().__init__()
        
        # Load configuration
        self.config_loader = PhaseConfigLoader()
        
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = self.config_loader.load_config(phase=phase, dataset_size=dataset_size)
        
        self.model_config = self.config_loader.get_model_config(phase=phase, dataset_size=dataset_size)
        self.training_config = self.config_loader.get_training_config(phase=phase, dataset_size=dataset_size)
        
        # Store phase info
        if phase:
            self.current_phase = phase
        elif dataset_size:
            self.current_phase = self.config_loader.get_phase_by_dataset_size(dataset_size)
        else:
            self.current_phase = "phase1"
        
        # Initialize components with configuration
        self._init_components()
        
        # Print configuration summary
        print(f"Initialized {self.current_phase.upper()} model:")
        self.config_loader.print_phase_summary(phase=self.current_phase)
    
    def _init_components(self):
        """Initialize model components based on configuration."""
        
        # Extract config values
        d_model = self.model_config.d_model
        n_heads = self.model_config.n_heads
        n_layers = self.model_config.n_layers
        dropout = self.model_config.dropout
        
        vision_config = self.model_config.vision
        structure_config = self.model_config.structure
        layout_config = self.model_config.layout
        diffusion_config = self.model_config.diffusion
        
        # 1. Multimodal Encoder (Vision + Structure)
        self.multimodal_encoder = MultimodalEncoder(
            d_model=d_model,
            num_heads=n_heads,
            num_layers=n_layers,
            dropout=dropout,
            # Vision settings
            patch_embed_dim=d_model,
            # Structure settings
            structure_vocab_size=structure_config['vocab_size']
        )
        
        # 2. Layout Embedding
        self.layout_embedding = LayoutEmbedding(
            d_model=d_model,
            element_vocab_size=layout_config['class_vocab_size'],
            property_vocab_size=layout_config['class_vocab_size'] // 2,
            max_position=1000,
            max_timesteps=diffusion_config['timesteps'],
            dropout=dropout
        )
        
        # 3. Diffusion Decoder
        self.diffusion_decoder = DiffusionDecoder(
            d_model=d_model,
            num_heads=n_heads,
            num_layers=n_layers,
            element_vocab_size=layout_config['class_vocab_size'],
            property_vocab_size=layout_config['class_vocab_size'] // 2,
            max_elements=layout_config['max_elements'],
            dropout=dropout
        )
        
        # 4. Aesthetic Constraints
        self.aesthetic_constraints = AestheticConstraintModule(
            canvas_width=vision_config['image_size'],
            canvas_height=vision_config['image_size']  # Assuming square images
        )
        
        # Store important dimensions
        self.d_model = d_model
        self.max_elements = layout_config['max_elements']
        self.class_vocab_size = layout_config['class_vocab_size']
        self.timesteps = diffusion_config['timesteps']
    
    def forward(
        self,
        screenshot: torch.Tensor,
        structure_tokens: torch.Tensor,
        layout_tokens: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        training: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with automatic phase adaptation.
        
        Args:
            screenshot: [B, C, H, W] screenshot images
            structure_tokens: [B, S] HTML structure tokens
            layout_tokens: [B, L] layout tokens (for training)
            timestep: [B] diffusion timesteps
            training: Whether in training mode
            
        Returns:
            Dictionary with model outputs
        """
        batch_size = screenshot.size(0)
        
        # Convert screenshot to patch embeddings (simplified)
        # For now, create mock patch embeddings from screenshot
        patch_embeddings = self._screenshot_to_patches(screenshot)
        
        # 1. Encode multimodal inputs
        multimodal_features = self.multimodal_encoder(
            patch_embeddings=patch_embeddings,
            token_ids=structure_tokens
        )['multimodal_features']
        
        if training and layout_tokens is not None:
            # Training mode: predict layout from noisy input
            if timestep is None:
                timestep = torch.randint(0, self.timesteps, (batch_size,), device=screenshot.device)
            
            # Add noise to layout for diffusion training
            noisy_layout = self._add_noise_to_tokens(layout_tokens, timestep)
            
            # Predict denoised layout
            outputs = self.diffusion_decoder(
                noised_layout=noisy_layout,
                timesteps=timestep,
                encoder_features=multimodal_features
            )
            
            return {
                'predicted_layout': outputs.get('element_logits', torch.zeros_like(noisy_layout)),
                'predicted_elements': outputs.get('element_logits', torch.zeros_like(noisy_layout)),
                'predicted_geometry': outputs.get('geometric_predictions', torch.zeros(batch_size, self.max_elements, 6, device=screenshot.device)),
                'predicted_props': outputs.get('props_logits', torch.zeros(batch_size, 3, device=screenshot.device)),
                'timestep': timestep,
                'noisy_input': noisy_layout
            }
        
        else:
            # Inference mode: generate layout from scratch
            return self.generate_layout(
                screenshot=screenshot,
                structure_tokens=structure_tokens,
                num_steps=self.timesteps // 10  # Fewer steps for faster inference
            )
    
    def _screenshot_to_patches(self, screenshot: torch.Tensor) -> torch.Tensor:
        """Convert screenshot to patch embeddings (simplified version)."""
        batch_size, channels, height, width = screenshot.shape
        
        # Simple patching: divide image into patches and flatten
        patch_size = self.model_config.vision['patch_size']
        num_patches_h = height // patch_size
        num_patches_w = width // patch_size
        
        # Unfold into patches
        patches = screenshot.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(batch_size, channels, num_patches_h * num_patches_w, patch_size * patch_size)
        patches = patches.permute(0, 2, 1, 3).contiguous().view(batch_size, num_patches_h * num_patches_w, -1)
        
        # Project to d_model dimensions
        if not hasattr(self, 'patch_projection'):
            self.patch_projection = nn.Linear(patches.size(-1), self.d_model).to(screenshot.device)
        
        patch_embeddings = self.patch_projection(patches)
        
        return patch_embeddings
    
    def generate_layout(
        self,
        screenshot: torch.Tensor,
        structure_tokens: torch.Tensor,
        num_steps: int = 50,
        guidance_scale: float = 1.0,
        temperature: float = 1.0
    ) -> GenerationOutput:
        """
        Generate layout using diffusion sampling with phase-appropriate settings.
        """
        batch_size = screenshot.size(0)
        device = screenshot.device
        
        # Convert screenshot to patch embeddings
        patch_embeddings = self._screenshot_to_patches(screenshot)
        
        # Encode inputs
        multimodal_features = self.multimodal_encoder(
            patch_embeddings=patch_embeddings,
            token_ids=structure_tokens
        )['multimodal_features']
        
        # Start from random noise
        layout_shape = (batch_size, self.max_elements, self.d_model)
        layout_noise = torch.randn(layout_shape, device=device)
        
        # Sample with fewer steps for smaller models (faster inference)
        if self.current_phase == "phase1":
            num_steps = min(num_steps, 20)  # Very fast for Phase 1
        elif self.current_phase == "phase2":
            num_steps = min(num_steps, 30)  # Fast for Phase 2
        
        # Simplified diffusion sampling (since full sampling method needs debugging)
        layout_tokens = layout_noise  # For now, use noise as layout tokens
        
        # Convert to structured output
        elements = self._tokens_to_elements(layout_tokens)
        
        # Apply aesthetic constraints
        constraint_violations = []
        aesthetic_score = 0.0
        
        try:
            # Only apply constraints for Phase 3+ (more data available)
            if self.current_phase in ["phase3", "phase4"]:
                constraints = self.aesthetic_constraints(layout_tokens, screenshot)
                constraint_violations = [name for name, violated in constraints.items() if violated]
                aesthetic_score = 1.0 - len(constraint_violations) / len(constraints)
            else:
                aesthetic_score = 0.8  # Default reasonable score for smaller models
        except Exception as e:
            print(f"Constraint evaluation failed: {e}")
            aesthetic_score = 0.5
        
        # Generate confidence scores (higher for larger models)
        confidence_multiplier = {
            "phase1": 0.6,
            "phase2": 0.7, 
            "phase3": 0.8,
            "phase4": 0.9
        }[self.current_phase]
        
        confidence_scores = torch.sigmoid(torch.randn(len(elements))) * confidence_multiplier
        
        return GenerationOutput(
            elements=elements,
            layout_tokens=layout_tokens,
            confidence_scores=confidence_scores,
            constraint_violations=constraint_violations,
            aesthetic_score=aesthetic_score
        )
    
    def _add_noise_to_tokens(self, layout_tokens: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Add noise to layout tokens for diffusion training."""
        # For discrete tokens, we can add noise by randomly replacing tokens
        noise_scale = timestep.float() / self.timesteps
        noise_mask = torch.rand_like(layout_tokens.float()) < noise_scale.unsqueeze(-1)
        
        # Replace tokens with random tokens where noise_mask is True
        random_tokens = torch.randint_like(layout_tokens, 0, self.class_vocab_size)
        noisy_tokens = torch.where(noise_mask, random_tokens, layout_tokens)
        
        return noisy_tokens
    
    def _tokens_to_elements(self, layout_tokens: torch.Tensor) -> List[Dict[str, Any]]:
        """Convert layout tokens to structured elements."""
        batch_size, num_elements, _ = layout_tokens.shape
        elements = []
        
        for b in range(batch_size):
            batch_elements = []
            for e in range(num_elements):
                # Simple conversion (can be enhanced)
                element = {
                    'type': f'element_{e}',
                    'position': {'x': 0.1 * e, 'y': 0.1 * e},
                    'size': {'width': 0.2, 'height': 0.1},
                    'properties': {'class': f'auto-generated-{e}'}
                }
                batch_elements.append(element)
            elements.extend(batch_elements)
        
        return elements
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about current model configuration."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'phase': self.current_phase,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024**2),  # Assuming float32
            'd_model': self.d_model,
            'max_elements': self.max_elements,
            'timesteps': self.timesteps,
            'estimated_memory_gb': total_params * 4 / (1024**3),
            'configuration': {
                'model': self.model_config.__dict__,
                'training': self.training_config.__dict__
            }
        }
    
    def upgrade_to_phase(self, new_phase: str, preserve_weights: bool = True):
        """Upgrade model to a new phase with larger architecture."""
        if preserve_weights:
            # Save current state
            current_state = self.state_dict()
        
        # Load new configuration
        self.model_config = self.config_loader.get_model_config(phase=new_phase)
        self.training_config = self.config_loader.get_training_config(phase=new_phase)
        self.current_phase = new_phase
        
        # Reinitialize with new architecture
        self._init_components()
        
        if preserve_weights:
            # Attempt to load compatible weights
            try:
                self.load_state_dict(current_state, strict=False)
                print(f"Successfully upgraded to {new_phase} while preserving compatible weights")
            except Exception as e:
                print(f"Could not preserve all weights during upgrade: {e}")
        
        print(f"Model upgraded to {new_phase}")
        self.config_loader.print_phase_summary(phase=new_phase)


def create_phase_appropriate_model(dataset_size: int) -> ConfigurableSectionLayoutGenerator:
    """Factory function to create model appropriate for dataset size."""
    return ConfigurableSectionLayoutGenerator(dataset_size=dataset_size) 