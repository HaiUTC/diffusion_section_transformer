"""
Aesthetic Constraint Module - Step 3: Model Architecture Implementation

This module implements:
- Differentiable Loss Layers for aesthetic constraints
- Overlap minimization (IoU-based collision detection)
- Alignment loss (grid alignment and visual harmony)
- Gradient guidance during sampling for designer-aligned outputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Optional, Tuple, List


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes
    
    Args:
        boxes1: [N, 4] boxes in format [x1, y1, x2, y2]
        boxes2: [M, 4] boxes in format [x1, y1, x2, y2]
        
    Returns:
        IoU matrix [N, M]
    """
    # Expand dimensions for broadcasting
    boxes1 = boxes1.unsqueeze(1)  # [N, 1, 4]
    boxes2 = boxes2.unsqueeze(0)  # [1, M, 4]
    
    # Compute intersection coordinates
    x1 = torch.max(boxes1[..., 0], boxes2[..., 0])  # [N, M]
    y1 = torch.max(boxes1[..., 1], boxes2[..., 1])  # [N, M]
    x2 = torch.min(boxes1[..., 2], boxes2[..., 2])  # [N, M]
    y2 = torch.min(boxes1[..., 3], boxes2[..., 3])  # [N, M]
    
    # Compute intersection area
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # Compute box areas
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])  # [N, 1]
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])  # [1, M]
    
    # Compute union area
    union = area1 + area2 - intersection
    
    # Compute IoU, avoiding division by zero
    iou = intersection / (union + 1e-6)
    
    return iou


class OverlapConstraint(nn.Module):
    """Overlap minimization constraint using IoU-based collision detection"""
    
    def __init__(self, overlap_threshold: float = 0.1, penalty_weight: float = 10.0):
        super().__init__()
        self.overlap_threshold = overlap_threshold
        self.penalty_weight = penalty_weight
        
    def forward(self, bounding_boxes: torch.Tensor, 
                element_types: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute overlap loss for layout elements
        
        Args:
            bounding_boxes: [batch, num_elements, 4] in format [x1, y1, x2, y2]
            element_types: [batch, num_elements] element type IDs (for type-specific rules)
            
        Returns:
            Overlap loss scalar
        """
        batch_size, num_elements, _ = bounding_boxes.shape
        total_loss = 0.0
        
        for batch_idx in range(batch_size):
            boxes = bounding_boxes[batch_idx]  # [num_elements, 4]
            
            # Filter out invalid boxes (zero area or padding)
            valid_mask = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            if valid_mask.sum() < 2:
                continue  # Skip if less than 2 valid boxes
                
            valid_boxes = boxes[valid_mask]
            
            # Compute pairwise IoU
            iou_matrix = box_iou(valid_boxes, valid_boxes)
            
            # Remove diagonal (self-intersection)
            n = valid_boxes.size(0)
            mask = ~torch.eye(n, device=iou_matrix.device, dtype=torch.bool)
            overlap_values = iou_matrix[mask]
            
            # Penalize overlaps above threshold
            overlap_penalty = torch.clamp(overlap_values - self.overlap_threshold, min=0)
            batch_loss = self.penalty_weight * overlap_penalty.sum()
            
            total_loss += batch_loss
        
        return total_loss / batch_size


class AlignmentConstraint(nn.Module):
    """Alignment constraint for grid alignment and visual harmony"""
    
    def __init__(self, grid_size: int = 8, alignment_weight: float = 1.0, 
                 spacing_weight: float = 0.5):
        super().__init__()
        self.grid_size = grid_size
        self.alignment_weight = alignment_weight
        self.spacing_weight = spacing_weight
        
    def forward(self, bounding_boxes: torch.Tensor,
                canvas_width: float = 1920, canvas_height: float = 1080) -> torch.Tensor:
        """
        Compute alignment loss for layout elements
        
        Args:
            bounding_boxes: [batch, num_elements, 4] in format [x1, y1, x2, y2]
            canvas_width: Canvas width for normalization
            canvas_height: Canvas height for normalization
            
        Returns:
            Alignment loss scalar
        """
        batch_size, num_elements, _ = bounding_boxes.shape
        
        # Normalize boxes to [0, 1] range
        normalized_boxes = bounding_boxes.clone()
        normalized_boxes[:, :, [0, 2]] /= canvas_width
        normalized_boxes[:, :, [1, 3]] /= canvas_height
        
        total_loss = 0.0
        
        for batch_idx in range(batch_size):
            boxes = normalized_boxes[batch_idx]  # [num_elements, 4]
            
            # Filter out invalid boxes
            valid_mask = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            if valid_mask.sum() < 2:
                continue
                
            valid_boxes = boxes[valid_mask]
            
            # Grid alignment loss
            grid_step = 1.0 / self.grid_size
            
            # Check alignment to grid lines
            x_coords = torch.cat([valid_boxes[:, 0], valid_boxes[:, 2]])  # Left and right edges
            y_coords = torch.cat([valid_boxes[:, 1], valid_boxes[:, 3]])  # Top and bottom edges
            
            # Distance to nearest grid line
            x_grid_dist = torch.min(x_coords % grid_step, grid_step - (x_coords % grid_step))
            y_grid_dist = torch.min(y_coords % grid_step, grid_step - (y_coords % grid_step))
            
            grid_loss = self.alignment_weight * (x_grid_dist.mean() + y_grid_dist.mean())
            
            # Spacing consistency loss (elements should have consistent gaps)
            centers_x = (valid_boxes[:, 0] + valid_boxes[:, 2]) / 2
            centers_y = (valid_boxes[:, 1] + valid_boxes[:, 3]) / 2
            
            # Pairwise distances
            dist_x = torch.abs(centers_x.unsqueeze(1) - centers_x.unsqueeze(0))
            dist_y = torch.abs(centers_y.unsqueeze(1) - centers_y.unsqueeze(0))
            
            # Remove diagonal
            n = centers_x.size(0)
            mask = ~torch.eye(n, device=dist_x.device, dtype=torch.bool)
            
            # Encourage consistent spacing (penalize variance in distances)
            spacing_x_var = torch.var(dist_x[mask])
            spacing_y_var = torch.var(dist_y[mask])
            spacing_loss = self.spacing_weight * (spacing_x_var + spacing_y_var)
            
            total_loss += grid_loss + spacing_loss
        
        return total_loss / batch_size


class ProportionConstraint(nn.Module):
    """Proportion constraint for aesthetic aspect ratios and sizing"""
    
    def __init__(self, golden_ratio_weight: float = 0.5, size_harmony_weight: float = 0.3):
        super().__init__()
        self.golden_ratio = (1 + math.sqrt(5)) / 2  # ~1.618
        self.golden_ratio_weight = golden_ratio_weight
        self.size_harmony_weight = size_harmony_weight
        
    def forward(self, bounding_boxes: torch.Tensor,
                element_types: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute proportion loss for layout elements
        
        Args:
            bounding_boxes: [batch, num_elements, 4] in format [x1, y1, x2, y2]
            element_types: [batch, num_elements] element type IDs
            
        Returns:
            Proportion loss scalar
        """
        batch_size, num_elements, _ = bounding_boxes.shape
        total_loss = 0.0
        
        for batch_idx in range(batch_size):
            boxes = bounding_boxes[batch_idx]  # [num_elements, 4]
            
            # Filter out invalid boxes
            valid_mask = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            if valid_mask.sum() == 0:
                continue
                
            valid_boxes = boxes[valid_mask]
            
            # Compute aspect ratios
            widths = valid_boxes[:, 2] - valid_boxes[:, 0]
            heights = valid_boxes[:, 3] - valid_boxes[:, 1]
            aspect_ratios = widths / (heights + 1e-6)
            
            # Golden ratio preference for certain elements
            golden_ratio_loss = torch.min(
                torch.abs(aspect_ratios - self.golden_ratio),
                torch.abs(aspect_ratios - 1/self.golden_ratio)
            ).mean()
            
            # Size harmony (similar elements should have similar sizes)
            areas = widths * heights
            area_var = torch.var(areas)
            size_harmony_loss = area_var / (areas.mean() + 1e-6)  # Normalized variance
            
            batch_loss = (self.golden_ratio_weight * golden_ratio_loss + 
                          self.size_harmony_weight * size_harmony_loss)
            total_loss += batch_loss
        
        return total_loss / batch_size


class ReadabilityConstraint(nn.Module):
    """Readability constraint for text elements and visual hierarchy"""
    
    def __init__(self, min_text_size: float = 12.0, hierarchy_weight: float = 1.0):
        super().__init__()
        self.min_text_size = min_text_size
        self.hierarchy_weight = hierarchy_weight
        
    def forward(self, bounding_boxes: torch.Tensor,
                element_types: torch.Tensor,
                text_element_ids: List[int] = [1, 2, 3]) -> torch.Tensor:  # heading, paragraph, etc.
        """
        Compute readability loss for text elements
        
        Args:
            bounding_boxes: [batch, num_elements, 4] in format [x1, y1, x2, y2]
            element_types: [batch, num_elements] element type IDs
            text_element_ids: List of element type IDs that are text elements
            
        Returns:
            Readability loss scalar
        """
        batch_size, num_elements, _ = bounding_boxes.shape
        total_loss = 0.0
        
        for batch_idx in range(batch_size):
            boxes = bounding_boxes[batch_idx]  # [num_elements, 4]
            types = element_types[batch_idx]   # [num_elements]
            
            # Filter text elements
            text_mask = torch.isin(types, torch.tensor(text_element_ids, device=types.device))
            if text_mask.sum() == 0:
                continue
                
            text_boxes = boxes[text_mask]
            text_types = types[text_mask]
            
            # Minimum size constraint
            heights = text_boxes[:, 3] - text_boxes[:, 1]
            min_size_violations = torch.clamp(self.min_text_size - heights, min=0)
            min_size_loss = min_size_violations.sum()
            
            # Hierarchy constraint (headings should be larger than paragraphs)
            if len(text_element_ids) > 1:
                hierarchy_loss = 0.0
                for i, type_id in enumerate(text_element_ids):
                    type_mask = (text_types == type_id)
                    if type_mask.sum() > 0:
                        type_heights = heights[type_mask]
                        # Higher priority elements (lower IDs) should be larger
                        expected_min_height = self.min_text_size * (len(text_element_ids) - i)
                        hierarchy_violations = torch.clamp(expected_min_height - type_heights, min=0)
                        hierarchy_loss += hierarchy_violations.mean()
            else:
                hierarchy_loss = 0.0
            
            batch_loss = min_size_loss + self.hierarchy_weight * hierarchy_loss
            total_loss += batch_loss
        
        return total_loss / batch_size


class AestheticConstraintModule(nn.Module):
    """
    Complete Aesthetic Constraint Module with differentiable loss layers
    
    Implements:
    - Overlap minimization: C_olp = ∑_{i≠j} IoU(b_i, b_j)
    - Alignment loss: C_alg = ||align_error||²
    - Proportion constraints for golden ratio and size harmony
    - Readability constraints for text elements
    """
    
    def __init__(self, 
                 overlap_threshold: float = 0.1,
                 overlap_weight: float = 10.0,
                 alignment_weight: float = 1.0,
                 proportion_weight: float = 0.5,
                 readability_weight: float = 1.5,
                 canvas_width: float = 1920,
                 canvas_height: float = 1080):
        super().__init__()
        
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        
        # Individual constraint modules
        self.overlap_constraint = OverlapConstraint(overlap_threshold, overlap_weight)
        self.alignment_constraint = AlignmentConstraint(alignment_weight=alignment_weight)
        self.proportion_constraint = ProportionConstraint()
        self.readability_constraint = ReadabilityConstraint()
        
        # Constraint weights
        self.overlap_weight = overlap_weight
        self.alignment_weight = alignment_weight
        self.proportion_weight = proportion_weight
        self.readability_weight = readability_weight
        
    def forward(self, bounding_boxes: torch.Tensor,
                element_types: Optional[torch.Tensor] = None,
                return_individual: bool = False) -> Dict[str, torch.Tensor]:
        """
        Compute aesthetic constraint losses
        
        Args:
            bounding_boxes: [batch, num_elements, 4] in format [x1, y1, x2, y2]
            element_types: [batch, num_elements] element type IDs
            return_individual: Whether to return individual constraint losses
            
        Returns:
            Dictionary containing constraint losses
        """
        # Compute individual constraints
        overlap_loss = self.overlap_constraint(bounding_boxes, element_types)
        alignment_loss = self.alignment_constraint(bounding_boxes, self.canvas_width, self.canvas_height)
        proportion_loss = self.proportion_constraint(bounding_boxes, element_types)
        
        # Readability loss (only if element types provided)
        if element_types is not None:
            readability_loss = self.readability_constraint(bounding_boxes, element_types)
        else:
            readability_loss = torch.tensor(0.0, device=bounding_boxes.device)
        
        # Total constraint loss
        total_loss = (self.overlap_weight * overlap_loss +
                     self.alignment_weight * alignment_loss +
                     self.proportion_weight * proportion_loss +
                     self.readability_weight * readability_loss)
        
        results = {
            'total_constraint_loss': total_loss,
            'C_olp': overlap_loss,
            'C_alg': alignment_loss
        }
        
        if return_individual:
            results.update({
                'overlap_loss': overlap_loss,
                'alignment_loss': alignment_loss,
                'proportion_loss': proportion_loss,
                'readability_loss': readability_loss
            })
        
        return results
    
    def gradient_guidance(self, layout_predictions: torch.Tensor,
                         element_types: Optional[torch.Tensor] = None,
                         guidance_strength: float = 0.1) -> torch.Tensor:
        """
        Apply gradient guidance for aesthetic refinement
        
        Args:
            layout_predictions: [batch, num_elements, 6] predictions (x, y, w, h, ...)
            element_types: [batch, num_elements] element type IDs
            guidance_strength: Strength of gradient guidance (λ parameter)
            
        Returns:
            Refined layout predictions
        """
        # Convert predictions to bounding boxes
        x, y, w, h = layout_predictions[:, :, 0], layout_predictions[:, :, 1], layout_predictions[:, :, 2], layout_predictions[:, :, 3]
        bounding_boxes = torch.stack([
            x, y, x + w, y + h
        ], dim=-1)
        
        # Compute constraint losses
        constraint_results = self.forward(bounding_boxes, element_types)
        total_constraint = constraint_results['total_constraint_loss']
        
        # Compute gradients with respect to layout predictions
        if layout_predictions.requires_grad:
            grad = torch.autograd.grad(
                total_constraint, layout_predictions,
                retain_graph=True, create_graph=True
            )[0]
            
            # Apply gradient guidance: x̂₀ ← x̂₀ - λ ∇_{x̂₀} (C_alg + C_olp)
            refined_predictions = layout_predictions - guidance_strength * grad
            
            return refined_predictions
        else:
            return layout_predictions


def create_aesthetic_constraint_config():
    """Create default configuration for aesthetic constraints"""
    return {
        'overlap_threshold': 0.1,
        'overlap_weight': 10.0,
        'alignment_weight': 1.0,
        'proportion_weight': 0.5,
        'readability_weight': 1.5,
        'canvas_width': 1920,
        'canvas_height': 1080
    }


def apply_aesthetic_guidance(model_output: torch.Tensor,
                           aesthetic_module: AestheticConstraintModule,
                           element_types: Optional[torch.Tensor] = None,
                           guidance_strength: float = 0.1,
                           num_refinement_steps: int = 3) -> torch.Tensor:
    """
    Apply iterative aesthetic guidance to model outputs
    
    Args:
        model_output: Raw model predictions
        aesthetic_module: Aesthetic constraint module
        element_types: Element type IDs
        guidance_strength: Gradient guidance strength
        num_refinement_steps: Number of refinement iterations
        
    Returns:
        Aesthetically refined predictions
    """
    refined_output = model_output
    
    for step in range(num_refinement_steps):
        refined_output = aesthetic_module.gradient_guidance(
            refined_output, element_types, guidance_strength
        )
        guidance_strength *= 0.8  # Decay guidance strength
    
    return refined_output 