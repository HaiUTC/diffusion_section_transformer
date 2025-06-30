"""
Dynamic Execution Optimization - DyDiT Implementation

This module implements Dynamic Diffusion Transformer (DyDiT) techniques for
adaptive computation along temporal and spatial dimensions:

- Timestep-wise Dynamic Width (TDW): Adjusts model width based on timesteps
- Spatial-wise Dynamic Token (SDT): Identifies simple patches for bypass
- Adaptive Computation Strategies: Dynamic resource allocation

Reference: Dynamic Diffusion Transformer optimization techniques
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import math
from dataclasses import dataclass


@dataclass
class DynamicConfig:
    """Configuration for dynamic execution optimization."""
    enable_timestep_dynamic_width: bool = True
    enable_spatial_dynamic_token: bool = True
    min_width_ratio: float = 0.25  # Minimum model width ratio
    max_width_ratio: float = 1.0   # Maximum model width ratio
    complexity_threshold: float = 0.3  # Threshold for bypass decisions
    adaptive_batch_size: bool = True
    dynamic_attention_heads: bool = True


class TimestepDynamicWidth:
    """
    Timestep-wise Dynamic Width (TDW) implementation.
    Adjusts model width based on generation timesteps - early steps use reduced
    capacity for coarse structure, later steps use full capacity for details.
    """
    
    def __init__(self, config: DynamicConfig):
        self.config = config
        self.width_schedule = self._create_width_schedule()
        
    def _create_width_schedule(self) -> Dict[str, float]:
        """
        Create timestep-based width scheduling.
        
        Returns:
            Dictionary mapping timestep ranges to width ratios
        """
        return {
            'early': self.config.min_width_ratio,     # 0-250 steps: coarse layout
            'middle': 0.6,                            # 250-750 steps: refinement  
            'late': self.config.max_width_ratio       # 750-1000 steps: details
        }
    
    def get_width_ratio(self, timestep: int, total_steps: int = 1000) -> float:
        """
        Get width ratio for given timestep.
        
        Args:
            timestep: Current diffusion timestep
            total_steps: Total number of diffusion steps
            
        Returns:
            Width ratio to use for this timestep
        """
        normalized_step = timestep / total_steps
        
        if normalized_step < 0.25:
            return self.width_schedule['early']
        elif normalized_step < 0.75:
            return self.width_schedule['middle']
        else:
            return self.width_schedule['late']
    
    def adapt_layer_width(self, layer: nn.Module, width_ratio: float) -> nn.Module:
        """
        Dynamically adapt layer width based on ratio.
        
        Args:
            layer: Neural network layer to adapt
            width_ratio: Width ratio to apply
            
        Returns:
            Adapted layer with reduced width
        """
        if isinstance(layer, nn.Linear):
            return self._adapt_linear_layer(layer, width_ratio)
        elif isinstance(layer, nn.MultiheadAttention):
            return self._adapt_attention_layer(layer, width_ratio)
        else:
            return layer  # Return unchanged for unsupported layers
    
    def _adapt_linear_layer(self, layer: nn.Linear, width_ratio: float) -> nn.Linear:
        """Adapt linear layer width."""
        original_out_features = layer.out_features
        new_out_features = int(original_out_features * width_ratio)
        
        # Create new layer with reduced width
        adapted_layer = nn.Linear(layer.in_features, new_out_features, 
                                bias=layer.bias is not None)
        
        # Copy relevant weights
        with torch.no_grad():
            adapted_layer.weight.data = layer.weight.data[:new_out_features, :]
            if layer.bias is not None:
                adapted_layer.bias.data = layer.bias.data[:new_out_features]
        
        return adapted_layer
    
    def _adapt_attention_layer(self, layer: nn.MultiheadAttention, 
                              width_ratio: float) -> nn.MultiheadAttention:
        """Adapt attention layer width."""
        original_heads = layer.num_heads
        new_heads = max(1, int(original_heads * width_ratio))
        
        # Ensure new_heads divides embed_dim evenly
        embed_dim = layer.embed_dim
        while embed_dim % new_heads != 0 and new_heads > 1:
            new_heads -= 1
            
        if new_heads != original_heads:
            adapted_layer = nn.MultiheadAttention(
                embed_dim, new_heads, dropout=layer.dropout,
                bias=layer.in_proj_bias is not None
            )
            # Copy relevant weights (simplified - full implementation would be more complex)
            return adapted_layer
        
        return layer


class SpatialDynamicToken:
    """
    Spatial-wise Dynamic Token (SDT) implementation.
    Identifies image patches and HTML elements where layout prediction is
    straightforward, allowing them to bypass computationally intensive blocks.
    """
    
    def __init__(self, config: DynamicConfig):
        self.config = config
        self.complexity_analyzer = ComplexityAnalyzer(config.complexity_threshold)
        
    def identify_simple_tokens(self, tokens: torch.Tensor, 
                              token_type: str = 'visual') -> torch.Tensor:
        """
        Identify tokens that can bypass complex processing.
        
        Args:
            tokens: Input tokens [batch, seq_len, embed_dim]
            token_type: Type of tokens ('visual' or 'structural')
            
        Returns:
            Boolean mask [batch, seq_len] - True for simple tokens
        """
        if token_type == 'visual':
            return self._identify_simple_visual_tokens(tokens)
        elif token_type == 'structural':
            return self._identify_simple_structural_tokens(tokens)
        else:
            return torch.zeros(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
    
    def _identify_simple_visual_tokens(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        """
        Identify simple visual patches (uniform backgrounds, simple patterns).
        
        Args:
            visual_tokens: Visual patch embeddings [batch, num_patches, embed_dim]
            
        Returns:
            Boolean mask for simple patches
        """
        batch_size, num_patches, embed_dim = visual_tokens.shape
        
        # Compute local variance as complexity measure
        token_variance = torch.var(visual_tokens, dim=-1)  # [batch, num_patches]
        
        # Compute spatial coherence (similarity to neighbors)
        spatial_coherence = self._compute_spatial_coherence(visual_tokens)
        
        # Simple tokens have low variance and high spatial coherence
        complexity_score = token_variance - spatial_coherence
        simple_mask = complexity_score < self.config.complexity_threshold
        
        return simple_mask
    
    def _identify_simple_structural_tokens(self, structural_tokens: torch.Tensor) -> torch.Tensor:
        """
        Identify simple structural elements (basic text blocks, standard containers).
        
        Args:
            structural_tokens: HTML structure embeddings [batch, num_tokens, embed_dim]
            
        Returns:
            Boolean mask for simple elements
        """
        batch_size, num_tokens, embed_dim = structural_tokens.shape
        
        # Compute embedding norm as complexity measure
        token_norms = torch.norm(structural_tokens, dim=-1)  # [batch, num_tokens]
        
        # Simple structural elements typically have standard embeddings
        norm_mean = torch.mean(token_norms, dim=-1, keepdim=True)
        norm_deviation = torch.abs(token_norms - norm_mean)
        
        simple_mask = norm_deviation < self.config.complexity_threshold
        
        return simple_mask
    
    def _compute_spatial_coherence(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial coherence for visual patches.
        
        Args:
            visual_tokens: Visual patch embeddings [batch, num_patches, embed_dim]
            
        Returns:
            Spatial coherence scores [batch, num_patches]
        """
        batch_size, num_patches, embed_dim = visual_tokens.shape
        
        # Assume patches are arranged in a grid (simplified)
        grid_size = int(math.sqrt(num_patches))
        if grid_size * grid_size != num_patches:
            # Fallback for non-square arrangements
            return torch.zeros(batch_size, num_patches, device=visual_tokens.device)
        
        # Reshape to spatial grid
        spatial_tokens = visual_tokens.view(batch_size, grid_size, grid_size, embed_dim)
        
        # Compute similarity with neighbors
        coherence_scores = torch.zeros(batch_size, grid_size, grid_size, device=visual_tokens.device)
        
        for i in range(grid_size):
            for j in range(grid_size):
                current_patch = spatial_tokens[:, i, j, :]  # [batch, embed_dim]
                neighbor_similarities = []
                
                # Check 4-connected neighbors
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < grid_size and 0 <= nj < grid_size:
                        neighbor_patch = spatial_tokens[:, ni, nj, :]
                        similarity = F.cosine_similarity(current_patch, neighbor_patch, dim=-1)
                        neighbor_similarities.append(similarity)
                
                if neighbor_similarities:
                    coherence_scores[:, i, j] = torch.stack(neighbor_similarities).mean(dim=0)
        
        # Flatten back to token sequence
        return coherence_scores.view(batch_size, num_patches)
    
    def apply_bypass_mask(self, tokens: torch.Tensor, bypass_mask: torch.Tensor,
                         bypass_output: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply bypass for simple tokens.
        
        Args:
            tokens: Input tokens [batch, seq_len, embed_dim]
            bypass_mask: Boolean mask for tokens to bypass [batch, seq_len]
            bypass_output: Optional precomputed output for bypassed tokens
            
        Returns:
            Tokens with bypass applied
        """
        if bypass_output is None:
            # Use identity transformation for bypassed tokens
            bypass_output = tokens.clone()
        
        # Apply bypass mask
        output_tokens = tokens.clone()
        output_tokens[bypass_mask.unsqueeze(-1).expand_as(tokens)] = \
            bypass_output[bypass_mask.unsqueeze(-1).expand_as(tokens)]
        
        return output_tokens


class ComplexityAnalyzer:
    """
    Analyzes computational complexity of different tokens and regions.
    """
    
    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
        
    def compute_complexity_score(self, tokens: torch.Tensor, 
                                token_type: str = 'visual') -> torch.Tensor:
        """
        Compute complexity scores for tokens.
        
        Args:
            tokens: Input tokens [batch, seq_len, embed_dim]
            token_type: Type of tokens ('visual' or 'structural')
            
        Returns:
            Complexity scores [batch, seq_len]
        """
        if token_type == 'visual':
            return self._visual_complexity(tokens)
        elif token_type == 'structural':
            return self._structural_complexity(tokens)
        else:
            return torch.ones(tokens.shape[:2], device=tokens.device)
    
    def _visual_complexity(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        """Compute visual complexity based on token variance and gradients."""
        # Token variance
        variance_score = torch.var(visual_tokens, dim=-1)
        
        # Gradient magnitude (simplified)
        grad_score = torch.norm(visual_tokens, dim=-1)
        
        # Combine scores
        complexity = 0.6 * variance_score + 0.4 * grad_score
        return complexity
    
    def _structural_complexity(self, structural_tokens: torch.Tensor) -> torch.Tensor:
        """Compute structural complexity based on embedding characteristics."""
        # Embedding magnitude
        magnitude_score = torch.norm(structural_tokens, dim=-1)
        
        # Entropy-like measure
        token_probs = F.softmax(structural_tokens, dim=-1)
        entropy_score = -torch.sum(token_probs * torch.log(token_probs + 1e-8), dim=-1)
        
        # Combine scores
        complexity = 0.5 * magnitude_score + 0.5 * entropy_score
        return complexity


class AdaptiveComputationStrategy:
    """
    Adaptive computation strategy that combines TDW and SDT for optimal resource allocation.
    """
    
    def __init__(self, config: DynamicConfig):
        self.config = config
        self.tdw = TimestepDynamicWidth(config)
        self.sdt = SpatialDynamicToken(config)
        
    def optimize_layer_computation(self, layer: nn.Module, 
                                  visual_tokens: torch.Tensor,
                                  structural_tokens: torch.Tensor,
                                  timestep: int,
                                  total_steps: int = 1000) -> Dict[str, Any]:
        """
        Apply adaptive computation optimization to a layer.
        
        Args:
            layer: Neural network layer to optimize
            visual_tokens: Visual patch embeddings
            structural_tokens: HTML structure embeddings  
            timestep: Current diffusion timestep
            total_steps: Total diffusion steps
            
        Returns:
            Optimization results and adapted layer
        """
        # Step 1: Determine width ratio based on timestep
        width_ratio = self.tdw.get_width_ratio(timestep, total_steps)
        
        # Step 2: Identify simple tokens for bypass
        visual_bypass_mask = self.sdt.identify_simple_tokens(visual_tokens, 'visual')
        structural_bypass_mask = self.sdt.identify_simple_tokens(structural_tokens, 'structural')
        
        # Step 3: Adapt layer width if needed
        adapted_layer = layer
        if width_ratio < 1.0 and self.config.enable_timestep_dynamic_width:
            adapted_layer = self.tdw.adapt_layer_width(layer, width_ratio)
        
        return {
            'adapted_layer': adapted_layer,
            'width_ratio': width_ratio,
            'visual_bypass_mask': visual_bypass_mask,
            'structural_bypass_mask': structural_bypass_mask,
            'computation_savings': self._estimate_savings(
                width_ratio, visual_bypass_mask, structural_bypass_mask
            )
        }
    
    def _estimate_savings(self, width_ratio: float, 
                         visual_bypass_mask: torch.Tensor,
                         structural_bypass_mask: torch.Tensor) -> Dict[str, float]:
        """
        Estimate computational savings from optimizations.
        
        Args:
            width_ratio: Width reduction ratio
            visual_bypass_mask: Bypass mask for visual tokens
            structural_bypass_mask: Bypass mask for structural tokens
            
        Returns:
            Dictionary with savings estimates
        """
        # Width reduction savings
        width_savings = 1.0 - width_ratio
        
        # Bypass savings
        visual_bypass_ratio = visual_bypass_mask.float().mean().item()
        structural_bypass_ratio = structural_bypass_mask.float().mean().item()
        
        # Combined savings (rough estimate)
        total_savings = width_savings + 0.3 * (visual_bypass_ratio + structural_bypass_ratio)
        total_savings = min(total_savings, 0.8)  # Cap at 80% savings
        
        return {
            'width_savings': width_savings,
            'visual_bypass_savings': visual_bypass_ratio,
            'structural_bypass_savings': structural_bypass_ratio,
            'total_estimated_savings': total_savings
        }


class DynamicExecutionOptimizer:
    """
    Main optimizer coordinating all dynamic execution techniques.
    """
    
    def __init__(self, config: DynamicConfig):
        self.config = config
        self.strategy = AdaptiveComputationStrategy(config)
        self.optimization_history = []
        
    def optimize_model_forward(self, model: nn.Module,
                              visual_tokens: torch.Tensor,
                              structural_tokens: torch.Tensor,
                              timestep: int,
                              total_steps: int = 1000) -> Dict[str, Any]:
        """
        Apply dynamic optimization to entire model forward pass.
        
        Args:
            model: Model to optimize
            visual_tokens: Visual patch embeddings
            structural_tokens: HTML structure embeddings
            timestep: Current diffusion timestep
            total_steps: Total diffusion steps
            
        Returns:
            Optimization results and metrics
        """
        optimization_results = {
            'optimized_layers': [],
            'total_savings': 0.0,
            'width_ratio': 1.0,
            'bypass_ratios': {'visual': 0.0, 'structural': 0.0}
        }
        
        # Apply optimization to each transformer block
        for name, module in model.named_modules():
            if isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
                layer_optimization = self.strategy.optimize_layer_computation(
                    module, visual_tokens, structural_tokens, timestep, total_steps
                )
                
                optimization_results['optimized_layers'].append({
                    'layer_name': name,
                    'optimization': layer_optimization
                })
                
                # Accumulate savings
                savings = layer_optimization['computation_savings']
                optimization_results['total_savings'] += savings['total_estimated_savings']
        
        # Average savings across layers
        if optimization_results['optimized_layers']:
            optimization_results['total_savings'] /= len(optimization_results['optimized_layers'])
        
        # Record optimization history
        self.optimization_history.append({
            'timestep': timestep,
            'savings': optimization_results['total_savings'],
            'timestamp': torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        })
        
        return optimization_results
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """
        Get statistics about optimization performance.
        
        Returns:
            Dictionary with optimization statistics
        """
        if not self.optimization_history:
            return {'message': 'No optimization history available'}
        
        savings_values = [record['savings'] for record in self.optimization_history]
        
        return {
            'total_optimizations': len(self.optimization_history),
            'average_savings': sum(savings_values) / len(savings_values),
            'max_savings': max(savings_values),
            'min_savings': min(savings_values),
            'config': self.config
        }


def create_dynamic_config(enable_all: bool = True, 
                         conservative: bool = False) -> DynamicConfig:
    """
    Create dynamic optimization configuration.
    
    Args:
        enable_all: Whether to enable all optimizations
        conservative: Whether to use conservative settings
        
    Returns:
        Configured DynamicConfig
    """
    if conservative:
        return DynamicConfig(
            enable_timestep_dynamic_width=enable_all,
            enable_spatial_dynamic_token=enable_all,
            min_width_ratio=0.5,  # More conservative
            max_width_ratio=1.0,
            complexity_threshold=0.5,  # Higher threshold
            adaptive_batch_size=False,
            dynamic_attention_heads=False
        )
    else:
        return DynamicConfig(
            enable_timestep_dynamic_width=enable_all,
            enable_spatial_dynamic_token=enable_all,
            min_width_ratio=0.25,  # Aggressive
            max_width_ratio=1.0,
            complexity_threshold=0.3,  # Lower threshold
            adaptive_batch_size=enable_all,
            dynamic_attention_heads=enable_all
        ) 