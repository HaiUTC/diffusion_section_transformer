"""
Loss Functions & Scheduling - Step 5 Implementation

This module implements comprehensive loss functions tailored to each training phase:
- Variance-aware loss scheduling for Phase 1 (small data)
- Element combination loss for @ concatenation syntax
- Modality-aware loss weighting for multimodal optimization
- Multi-scale consistency loss
- Production-ready multi-task loss functions

Reference: Step 5 specifications from instruction.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
import math
from collections import deque
import numpy as np


class ElementCombinationLoss(nn.Module):
    """
    Specialized cross-entropy loss for @ concatenation syntax in layout generation.
    
    Handles compound elements like "wrapper@div.class1@div.class2" by learning
    the mapping from visual+structural features to combined element types.
    """
    
    def __init__(self, element_vocab_size: int, regularization_weight: float = 1e-4):
        super().__init__()
        self.element_vocab_size = element_vocab_size
        self.regularization_weight = regularization_weight
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=0.1)
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                visual_features: Optional[torch.Tensor] = None,
                structure_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute element combination loss.
        
        Args:
            predictions: Predicted element logits [batch, seq_len, vocab_size]
            targets: Target element tokens [batch, seq_len] 
            visual_features: Visual feature context [batch, num_patches, d_model]
            structure_context: Structure feature context [batch, num_tokens, d_model]
            
        Returns:
            Element combination loss
        """
        batch_size, seq_len, vocab_size = predictions.shape
        
        # Reshape for cross-entropy computation
        predictions_flat = predictions.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        
        # Basic cross-entropy loss with label smoothing
        ce_loss = self.cross_entropy(predictions_flat, targets_flat)
        
        # Add contextual regularization if features available
        regularization_loss = 0.0
        
        if visual_features is not None and structure_context is not None:
            # Encourage coherence between visual and structural predictions
            visual_attention = torch.softmax(predictions.mean(dim=1), dim=-1)  # [batch, vocab_size]
            
            # Compute visual-structural alignment
            alignment_loss = self._compute_alignment_loss(
                visual_features, structure_context, visual_attention
            )
            regularization_loss += self.regularization_weight * alignment_loss
        
        # L2 regularization on predictions to prevent overconfidence
        l2_reg = self.regularization_weight * torch.norm(predictions, p=2)
        
        total_loss = ce_loss + regularization_loss + l2_reg
        
        return total_loss
    
    def _compute_alignment_loss(self, visual_features: torch.Tensor,
                               structure_context: torch.Tensor,
                               attention_weights: torch.Tensor) -> torch.Tensor:
        """Compute visual-structural alignment loss."""
        # Simplified alignment computation
        visual_pooled = visual_features.mean(dim=1)  # [batch, d_model]
        structure_pooled = structure_context.mean(dim=1)  # [batch, d_model]
        
        # Cosine similarity between modalities
        visual_norm = F.normalize(visual_pooled, p=2, dim=-1)
        structure_norm = F.normalize(structure_pooled, p=2, dim=-1)
        
        similarity = torch.sum(visual_norm * structure_norm, dim=-1)  # [batch]
        
        # Encourage high similarity weighted by attention
        attention_entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1)
        alignment_loss = -torch.mean(similarity * attention_entropy)
        
        return alignment_loss


class VarianceAwareLossScheduler(nn.Module):
    """
    Variance-aware loss scheduling for Phase 1 small data scenarios.
    
    Dynamically adjusts loss weighting based on statistical variability in
    alignment predictions, particularly effective when standard contrastive
    learning struggles with limited data.
    """
    
    def __init__(self, base_loss: nn.Module, variance_window: int = 100,
                 adaptive_weighting: bool = True):
        super().__init__()
        self.base_loss = base_loss
        self.variance_window = variance_window
        self.adaptive_weighting = adaptive_weighting
        
        # Variance tracking buffers
        self.register_buffer('loss_history', torch.zeros(variance_window))
        self.register_buffer('alignment_history', torch.zeros(variance_window))
        self.step_counter = 0
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                timesteps: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute variance-aware scheduled loss.
        
        L_total = α(t) * L_diffusion + β(t) * L_alignment + γ * L_regularization
        
        Where α(t) and β(t) are time-dependent weights based on prediction variance.
        """
        # Compute base losses
        base_loss_value = self.base_loss(predictions, targets, **kwargs)
        
        # Compute alignment loss if multimodal features available
        alignment_loss = 0.0
        if 'visual_features' in kwargs and 'structure_context' in kwargs:
            alignment_loss = self._compute_cross_modal_alignment(
                kwargs['visual_features'], kwargs['structure_context'], predictions
            )
        
        # Compute variance-based weights
        alpha_t, beta_t = self._compute_variance_weights(
            base_loss_value.detach(), alignment_loss
        )
        
        # Regularization loss
        gamma = 1e-4
        reg_loss = gamma * torch.norm(predictions, p=2)
        
        # Combined loss with time-dependent weighting
        total_loss = alpha_t * base_loss_value + beta_t * alignment_loss + reg_loss
        
        # Update history
        self._update_history(base_loss_value.detach(), alignment_loss)
        
        return {
            'total_loss': total_loss,
            'base_loss': base_loss_value,
            'alignment_loss': alignment_loss,
            'alpha_weight': alpha_t,
            'beta_weight': beta_t,
            'regularization_loss': reg_loss
        }
    
    def _compute_variance_weights(self, current_loss: torch.Tensor,
                                 alignment_loss: float) -> Tuple[float, float]:
        """Compute time-dependent weights based on prediction variance."""
        if self.step_counter < self.variance_window or not self.adaptive_weighting:
            return 1.0, 0.5  # Default weights
        
        # Compute variance from recent history
        loss_variance = torch.var(self.loss_history).item()
        alignment_variance = torch.var(self.alignment_history).item()
        
        # Adaptive weight computation
        # High variance → increase regularization, reduce base loss weight
        # Low variance → increase base loss weight, reduce regularization
        
        variance_ratio = loss_variance / (loss_variance + alignment_variance + 1e-8)
        
        alpha_t = 0.5 + 0.5 * (1.0 - variance_ratio)  # Range: [0.5, 1.0]
        beta_t = 0.2 + 0.8 * variance_ratio  # Range: [0.2, 1.0]
        
        return alpha_t, beta_t
    
    def _update_history(self, loss_value: torch.Tensor, alignment_loss: float):
        """Update loss history for variance tracking."""
        idx = self.step_counter % self.variance_window
        self.loss_history[idx] = loss_value.item()
        self.alignment_history[idx] = alignment_loss
        self.step_counter += 1
    
    def _compute_cross_modal_alignment(self, visual_features: torch.Tensor,
                                      structure_context: torch.Tensor,
                                      predictions: torch.Tensor) -> torch.Tensor:
        """Compute cross-modal alignment loss."""
        # Pool features
        visual_pooled = visual_features.mean(dim=1)
        structure_pooled = structure_context.mean(dim=1)
        
        # Compute prediction-guided attention
        pred_attention = torch.softmax(predictions.mean(dim=1), dim=-1)
        
        # Cross-modal contrastive loss
        visual_norm = F.normalize(visual_pooled, p=2, dim=-1)
        structure_norm = F.normalize(structure_pooled, p=2, dim=-1)
        
        # Cosine similarity matrix
        similarity_matrix = torch.matmul(visual_norm, structure_norm.transpose(0, 1))
        
        # Contrastive loss with prediction attention weighting
        batch_size = visual_features.size(0)
        labels = torch.arange(batch_size, device=visual_features.device)
        
        contrastive_loss = F.cross_entropy(similarity_matrix, labels)
        
        # Weight by prediction entropy (high entropy = uncertain predictions)
        pred_entropy = -torch.sum(pred_attention * torch.log(pred_attention + 1e-8), dim=-1)
        weighted_loss = torch.mean(contrastive_loss * pred_entropy.mean())
        
        return weighted_loss


class ModalityAwareLossWeighting(nn.Module):
    """
    Modality-aware loss weighting that dynamically balances contributions
    of each modality based on uncertainty or alignment quality.
    
    Particularly effective for low-data regimes where modality imbalance
    can bias the model.
    """
    
    def __init__(self, visual_weight: float = 0.4, structural_weight: float = 0.4,
                 geometric_weight: float = 0.2, aesthetic_weight: float = 0.0,
                 uncertainty_based: bool = True):
        super().__init__()
        self.visual_weight = visual_weight
        self.structural_weight = structural_weight
        self.geometric_weight = geometric_weight
        self.aesthetic_weight = aesthetic_weight
        self.uncertainty_based = uncertainty_based
        
        # Learnable modality weights if uncertainty-based
        if uncertainty_based:
            self.weight_predictor = nn.Sequential(
                nn.Linear(512, 256),  # Assuming 512-dim feature input
                nn.ReLU(),
                nn.Linear(256, 4),    # 4 modality weights
                nn.Softmax(dim=-1)
            )
    
    def forward(self, visual_loss: torch.Tensor, structural_loss: torch.Tensor,
                geometric_loss: torch.Tensor, aesthetic_loss: torch.Tensor = None,
                visual_features: Optional[torch.Tensor] = None,
                structural_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute modality-aware weighted loss.
        
        Args:
            visual_loss: Loss from visual modality
            structural_loss: Loss from structural modality  
            geometric_loss: Loss from geometric predictions
            aesthetic_loss: Loss from aesthetic constraints
            visual_features: Visual features for uncertainty estimation
            structural_features: Structural features for uncertainty estimation
            
        Returns:
            Weighted combined loss
        """
        if self.uncertainty_based and visual_features is not None and structural_features is not None:
            # Compute dynamic weights based on feature uncertainty
            weights = self._compute_uncertainty_weights(visual_features, structural_features)
            visual_w, structural_w, geometric_w, aesthetic_w = weights
        else:
            # Use fixed weights
            visual_w = self.visual_weight
            structural_w = self.structural_weight
            geometric_w = self.geometric_weight
            aesthetic_w = self.aesthetic_weight
        
        # Combine losses with dynamic weights
        total_loss = (visual_w * visual_loss + 
                     structural_w * structural_loss + 
                     geometric_w * geometric_loss)
        
        if aesthetic_loss is not None:
            total_loss += aesthetic_w * aesthetic_loss
        
        return total_loss
    
    def _compute_uncertainty_weights(self, visual_features: torch.Tensor,
                                   structural_features: torch.Tensor) -> torch.Tensor:
        """Compute uncertainty-based modality weights."""
        # Compute feature uncertainty (simplified as variance)
        visual_uncertainty = torch.var(visual_features, dim=1).mean(dim=1)  # [batch]
        structural_uncertainty = torch.var(structural_features, dim=1).mean(dim=1)  # [batch]
        
        # Combine uncertainties as input to weight predictor
        uncertainty_features = torch.cat([
            visual_features.mean(dim=1),      # Visual pooling
            structural_features.mean(dim=1),  # Structural pooling
            visual_uncertainty.unsqueeze(-1), # Visual uncertainty
            structural_uncertainty.unsqueeze(-1)  # Structural uncertainty
        ], dim=-1)
        
        # Predict weights [batch, 4]
        weights = self.weight_predictor(uncertainty_features)
        
        # Return batch-averaged weights
        return weights.mean(dim=0)


class MultiScaleConsistencyLoss(nn.Module):
    """
    Multi-scale consistency loss: L_consistency = Σ MSE(Layout_scale_i, Layout_scale_j)
    across different image resolutions to ensure scale-invariant layout generation.
    """
    
    def __init__(self, scales: List[int] = [256, 512, 768]):
        super().__init__()
        self.scales = scales
        self.mse_loss = nn.MSELoss()
        
    def forward(self, layout_predictions: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Compute multi-scale consistency loss.
        
        Args:
            layout_predictions: Dict mapping scales to layout predictions
                               {256: [batch, seq_len, d_model], 512: [...], ...}
            
        Returns:
            Multi-scale consistency loss
        """
        if len(layout_predictions) < 2:
            return torch.tensor(0.0, device=next(iter(layout_predictions.values())).device)
        
        total_loss = 0.0
        num_pairs = 0
        
        # Compare all pairs of scales
        scale_list = list(layout_predictions.keys())
        for i in range(len(scale_list)):
            for j in range(i + 1, len(scale_list)):
                scale_i, scale_j = scale_list[i], scale_list[j]
                pred_i = layout_predictions[scale_i]
                pred_j = layout_predictions[scale_j]
                
                # Ensure same sequence length (pad or truncate if needed)
                min_seq_len = min(pred_i.size(1), pred_j.size(1))
                pred_i = pred_i[:, :min_seq_len, :]
                pred_j = pred_j[:, :min_seq_len, :]
                
                # Compute MSE between scale predictions
                consistency_loss = self.mse_loss(pred_i, pred_j)
                total_loss += consistency_loss
                num_pairs += 1
        
        return total_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0)


class MultiTaskLossFunction(nn.Module):
    """
    Comprehensive multi-task loss function supporting all training phases.
    
    L_total = L_diffusion + λ₁*L_aesthetic + λ₂*L_alignment + λ₃*L_diversity + λ₄*L_props
    
    With optional dynamic weighting and phase-specific configurations.
    """
    
    def __init__(self, enable_aesthetic_loss: bool = True,
                 enable_alignment_loss: bool = True,
                 enable_diversity_loss: bool = True,
                 enable_props_loss: bool = True,
                 label_smoothing: float = 0.0,
                 guidance_scale: float = 7.5,
                 modality_weighter: Optional[ModalityAwareLossWeighting] = None,
                 consistency_loss: Optional[MultiScaleConsistencyLoss] = None,
                 dynamic_weighting: bool = False,
                 gradient_clipping: Optional[float] = None):
        super().__init__()
        
        self.enable_aesthetic_loss = enable_aesthetic_loss
        self.enable_alignment_loss = enable_alignment_loss
        self.enable_diversity_loss = enable_diversity_loss
        self.enable_props_loss = enable_props_loss
        self.guidance_scale = guidance_scale
        self.modality_weighter = modality_weighter
        self.consistency_loss = consistency_loss
        self.dynamic_weighting = dynamic_weighting
        self.gradient_clipping = gradient_clipping
        
        # Base loss functions
        self.diffusion_loss = nn.MSELoss()  # For noise prediction
        self.element_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.props_loss = nn.BCEWithLogitsLoss()  # For background properties
        
        # Dynamic weight tracking
        if dynamic_weighting:
            self.register_buffer('loss_weights', torch.ones(5))  # 5 loss components
            self.weight_momentum = 0.9
    
    def forward(self, model_outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                visual_features: Optional[torch.Tensor] = None,
                structural_features: Optional[torch.Tensor] = None,
                aesthetic_constraints: Optional[Any] = None) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive multi-task loss.
        
        Args:
            model_outputs: Dictionary containing model predictions
            targets: Dictionary containing ground truth targets  
            visual_features: Visual features for modality weighting
            structural_features: Structural features for modality weighting
            aesthetic_constraints: Aesthetic constraint module
            
        Returns:
            Dictionary with loss components and total loss
        """
        losses = {}
        
        # 1. Diffusion Loss (noise prediction)
        if 'noise_prediction' in model_outputs and 'noise_target' in targets:
            diffusion_loss = self.diffusion_loss(
                model_outputs['noise_prediction'], targets['noise_target']
            )
            losses['diffusion_loss'] = diffusion_loss
        
        # 2. Element Prediction Loss
        if 'element_logits' in model_outputs and 'element_targets' in targets:
            element_loss = self.element_loss(
                model_outputs['element_logits'].view(-1, model_outputs['element_logits'].size(-1)),
                targets['element_targets'].view(-1)
            )
            losses['element_loss'] = element_loss
        
        # 3. Geometric Prediction Loss  
        if 'geometric_predictions' in model_outputs and 'geometric_targets' in targets:
            geometric_loss = F.mse_loss(
                model_outputs['geometric_predictions'], targets['geometric_targets']
            )
            losses['geometric_loss'] = geometric_loss
        
        # 4. Props Loss (background properties)
        if self.enable_props_loss and 'props_logits' in model_outputs and 'props_targets' in targets:
            props_loss = self.props_loss(
                model_outputs['props_logits'], targets['props_targets'].float()
            )
            losses['props_loss'] = props_loss
        
        # 5. Aesthetic Loss
        if self.enable_aesthetic_loss and aesthetic_constraints is not None:
            if 'geometric_predictions' in model_outputs and 'element_logits' in model_outputs:
                geometric_preds = model_outputs['geometric_predictions']
                element_types = torch.argmax(model_outputs['element_logits'], dim=-1)
                
                # Convert geometric predictions to bounding boxes
                x, y, w, h = geometric_preds[:, :, 0], geometric_preds[:, :, 1], geometric_preds[:, :, 2], geometric_preds[:, :, 3]
                bounding_boxes = torch.stack([x, y, x + w, y + h], dim=-1)
                
                aesthetic_loss = aesthetic_constraints(bounding_boxes, element_types)
                losses['aesthetic_loss'] = aesthetic_loss
        
        # 6. Alignment Loss (cross-modal)
        if self.enable_alignment_loss and visual_features is not None and structural_features is not None:
            alignment_loss = self._compute_alignment_loss(visual_features, structural_features)
            losses['alignment_loss'] = alignment_loss
        
        # 7. Diversity Loss (encourage layout variety)
        if self.enable_diversity_loss and 'element_logits' in model_outputs:
            diversity_loss = self._compute_diversity_loss(model_outputs['element_logits'])
            losses['diversity_loss'] = diversity_loss
        
        # 8. Multi-scale Consistency Loss
        if self.consistency_loss is not None and 'multi_scale_predictions' in model_outputs:
            consistency_loss = self.consistency_loss(model_outputs['multi_scale_predictions'])
            losses['consistency_loss'] = consistency_loss
        
        # Combine losses with appropriate weighting
        total_loss = self._combine_losses(losses, visual_features, structural_features)
        
        losses['total_loss'] = total_loss
        return losses
    
    def _combine_losses(self, losses: Dict[str, torch.Tensor],
                       visual_features: Optional[torch.Tensor],
                       structural_features: Optional[torch.Tensor]) -> torch.Tensor:
        """Combine individual losses with appropriate weighting."""
        
        # Use modality-aware weighting if available
        if self.modality_weighter is not None:
            visual_loss = losses.get('diffusion_loss', torch.tensor(0.0))
            structural_loss = losses.get('element_loss', torch.tensor(0.0))
            geometric_loss = losses.get('geometric_loss', torch.tensor(0.0))
            aesthetic_loss = losses.get('aesthetic_loss', torch.tensor(0.0))
            
            weighted_loss = self.modality_weighter(
                visual_loss, structural_loss, geometric_loss, aesthetic_loss,
                visual_features, structural_features
            )
        else:
            # Standard loss combination
            weighted_loss = torch.tensor(0.0, device=next(iter(losses.values())).device)
            
            # Primary losses
            if 'diffusion_loss' in losses:
                weighted_loss += losses['diffusion_loss']
            if 'element_loss' in losses:
                weighted_loss += losses['element_loss']
            if 'geometric_loss' in losses:
                weighted_loss += 0.5 * losses['geometric_loss']
        
        # Add auxiliary losses
        if 'props_loss' in losses:
            weighted_loss += 0.2 * losses['props_loss']
        if 'aesthetic_loss' in losses:
            weighted_loss += 0.1 * losses['aesthetic_loss']
        if 'alignment_loss' in losses:
            weighted_loss += 0.3 * losses['alignment_loss']
        if 'diversity_loss' in losses:
            weighted_loss += 0.1 * losses['diversity_loss']
        if 'consistency_loss' in losses:
            weighted_loss += 0.2 * losses['consistency_loss']
        
        return weighted_loss
    
    def _compute_alignment_loss(self, visual_features: torch.Tensor,
                               structural_features: torch.Tensor) -> torch.Tensor:
        """Compute cross-modal alignment loss."""
        # Pool features
        visual_pooled = visual_features.mean(dim=1)
        structural_pooled = structural_features.mean(dim=1)
        
        # Contrastive loss between modalities
        visual_norm = F.normalize(visual_pooled, p=2, dim=-1)
        structural_norm = F.normalize(structural_pooled, p=2, dim=-1)
        
        similarity = torch.sum(visual_norm * structural_norm, dim=-1)
        alignment_loss = -torch.mean(similarity)  # Maximize similarity
        
        return alignment_loss
    
    def _compute_diversity_loss(self, element_logits: torch.Tensor) -> torch.Tensor:
        """Compute diversity loss to encourage layout variety within batches."""
        batch_size, seq_len, vocab_size = element_logits.shape
        
        # Compute element distribution across batch
        element_probs = torch.softmax(element_logits, dim=-1)
        batch_distribution = element_probs.mean(dim=(0, 1))  # [vocab_size]
        
        # Encourage uniform distribution (high entropy)
        entropy = -torch.sum(batch_distribution * torch.log(batch_distribution + 1e-8))
        max_entropy = torch.log(torch.tensor(vocab_size, dtype=torch.float))
        
        diversity_loss = max_entropy - entropy  # Minimize to maximize entropy
        
        return diversity_loss


def create_phase_loss_function(phase: str) -> nn.Module:
    """
    Factory function to create appropriate loss function for each training phase.
    
    Args:
        phase: Training phase ("phase1", "phase2", "phase3", "phase4")
        
    Returns:
        Configured loss function for the phase
    """
    if phase == "phase1":
        # Phase 1: Variance-aware loss with high regularization
        base_loss = ElementCombinationLoss(element_vocab_size=200, regularization_weight=1e-4)
        return VarianceAwareLossScheduler(base_loss=base_loss, adaptive_weighting=True)
    
    elif phase == "phase2":
        # Phase 2: Multi-task loss with modality-aware weighting
        modality_weighter = ModalityAwareLossWeighting(
            visual_weight=0.4, structural_weight=0.4, geometric_weight=0.2
        )
        consistency_loss = MultiScaleConsistencyLoss(scales=[256, 512, 768])
        
        return MultiTaskLossFunction(
            enable_aesthetic_loss=True,
            enable_alignment_loss=True,
            enable_diversity_loss=True,
            enable_props_loss=True,
            modality_weighter=modality_weighter,
            consistency_loss=consistency_loss
        )
    
    elif phase == "phase3":
        # Phase 3: Standard multi-task loss with label smoothing
        return MultiTaskLossFunction(
            enable_aesthetic_loss=True,
            enable_alignment_loss=True,
            enable_diversity_loss=True,
            enable_props_loss=True,
            label_smoothing=0.1,
            guidance_scale=7.5
        )
    
    elif phase == "phase4":
        # Phase 4: Production-ready loss with dynamic weighting
        modality_weighter = ModalityAwareLossWeighting(
            visual_weight=0.35, structural_weight=0.35, geometric_weight=0.2,
            aesthetic_weight=0.1, uncertainty_based=True
        )
        
        return MultiTaskLossFunction(
            enable_aesthetic_loss=True,
            enable_alignment_loss=True,
            enable_diversity_loss=True,
            enable_props_loss=True,
            modality_weighter=modality_weighter,
            dynamic_weighting=True,
            gradient_clipping=1.0
        )
    
    else:
        raise ValueError(f"Unknown phase: {phase}") 