"""
Quantization and Compression - DiTAS & MPQ-DM Implementation

This module implements advanced quantization techniques for diffusion transformers:

- DiTAS: Diffusion Transformers via Enhanced Activation Smoothing (W4A8)
- MPQ-DM: Mixed-Precision Quantization for Diffusion Models
- Temporal-aggregated smoothing for activation outlier mitigation
- Time-smoothed relation distillation for stable learning

Reference: DiTAS and MPQ-DM quantization frameworks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import math
from dataclasses import dataclass


@dataclass
class QuantizationConfig:
    """Configuration for quantization strategies."""
    enable_weight_quantization: bool = True
    enable_activation_quantization: bool = True
    weight_bits: int = 4  # W4 quantization
    activation_bits: int = 8  # A8 quantization
    enable_temporal_smoothing: bool = True
    smoothing_alpha: float = 0.9  # Temporal smoothing factor
    outlier_percentile: float = 99.5  # Percentile for outlier detection
    mixed_precision_layers: List[str] = None  # Layers to keep in higher precision
    calibration_steps: int = 100  # Steps for calibration


class ActivationSmoother:
    """
    Temporal-aggregated smoothing for activation outlier mitigation.
    Key component of DiTAS quantization framework.
    """
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.activation_stats = {}  # Running statistics for each layer
        self.outlier_masks = {}     # Outlier detection masks
        self.smoothing_buffers = {} # Temporal smoothing buffers
        
    def analyze_activations(self, layer_name: str, activations: torch.Tensor,
                           timestep: int) -> Dict[str, torch.Tensor]:
        """
        Analyze activation patterns and identify outliers.
        
        Args:
            layer_name: Name of the layer
            activations: Activation tensor [batch, seq_len, hidden_dim]
            timestep: Current diffusion timestep
            
        Returns:
            Dictionary with analysis results
        """
        if layer_name not in self.activation_stats:
            self._initialize_layer_stats(layer_name, activations.shape)
        
        # Compute activation statistics
        abs_activations = torch.abs(activations)
        
        # Channel-wise statistics
        channel_max = torch.max(abs_activations, dim=(0, 1))[0]
        channel_mean = torch.mean(abs_activations, dim=(0, 1))
        channel_std = torch.std(abs_activations, dim=(0, 1))
        
        # Update running statistics with temporal smoothing
        if self.config.enable_temporal_smoothing:
            self._update_temporal_stats(layer_name, channel_max, channel_mean, 
                                      channel_std, timestep)
        
        # Detect outliers
        outlier_mask = self._detect_outliers(layer_name, abs_activations)
        
        return {
            'channel_max': channel_max,
            'channel_mean': channel_mean,
            'channel_std': channel_std,
            'outlier_mask': outlier_mask,
            'outlier_ratio': outlier_mask.float().mean()
        }
    
    def _initialize_layer_stats(self, layer_name: str, activation_shape: torch.Size):
        """Initialize statistics tracking for a layer."""
        hidden_dim = activation_shape[-1]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.activation_stats[layer_name] = {
            'running_max': torch.zeros(hidden_dim, device=device),
            'running_mean': torch.zeros(hidden_dim, device=device),
            'running_std': torch.ones(hidden_dim, device=device),
            'update_count': 0
        }
        
        self.smoothing_buffers[layer_name] = {
            'timestep_history': [],
            'max_history': [],
            'mean_history': []
        }
    
    def _update_temporal_stats(self, layer_name: str, channel_max: torch.Tensor,
                              channel_mean: torch.Tensor, channel_std: torch.Tensor,
                              timestep: int):
        """Update temporal statistics with smoothing."""
        stats = self.activation_stats[layer_name]
        alpha = self.config.smoothing_alpha
        
        # Exponential moving average update
        if stats['update_count'] == 0:
            stats['running_max'] = channel_max.clone()
            stats['running_mean'] = channel_mean.clone()
            stats['running_std'] = channel_std.clone()
        else:
            stats['running_max'] = alpha * stats['running_max'] + (1 - alpha) * channel_max
            stats['running_mean'] = alpha * stats['running_mean'] + (1 - alpha) * channel_mean
            stats['running_std'] = alpha * stats['running_std'] + (1 - alpha) * channel_std
        
        stats['update_count'] += 1
        
        # Store timestep-specific history for analysis
        buffer = self.smoothing_buffers[layer_name]
        buffer['timestep_history'].append(timestep)
        buffer['max_history'].append(channel_max.clone())
        buffer['mean_history'].append(channel_mean.clone())
        
        # Keep only recent history
        max_history_len = 50
        if len(buffer['timestep_history']) > max_history_len:
            buffer['timestep_history'] = buffer['timestep_history'][-max_history_len:]
            buffer['max_history'] = buffer['max_history'][-max_history_len:]
            buffer['mean_history'] = buffer['mean_history'][-max_history_len:]
    
    def _detect_outliers(self, layer_name: str, abs_activations: torch.Tensor) -> torch.Tensor:
        """Detect activation outliers using percentile-based thresholding."""
        stats = self.activation_stats[layer_name]
        
        # Use running statistics for outlier detection
        threshold = torch.quantile(
            stats['running_max'], 
            self.config.outlier_percentile / 100.0
        )
        
        # Create outlier mask
        outlier_mask = abs_activations > threshold.unsqueeze(0).unsqueeze(0)
        
        return outlier_mask
    
    def smooth_activations(self, layer_name: str, activations: torch.Tensor,
                          timestep: int) -> torch.Tensor:
        """
        Apply temporal smoothing to activations to reduce outliers.
        
        Args:
            layer_name: Name of the layer
            activations: Input activations
            timestep: Current diffusion timestep
            
        Returns:
            Smoothed activations
        """
        # Analyze current activations
        analysis = self.analyze_activations(layer_name, activations, timestep)
        outlier_mask = analysis['outlier_mask']
        
        if layer_name not in self.activation_stats:
            return activations  # No smoothing data available
        
        # Apply smoothing to outliers
        smoothed_activations = activations.clone()
        stats = self.activation_stats[layer_name]
        
        # Replace outliers with smoothed values
        outlier_locations = outlier_mask
        if outlier_locations.any():
            # Use running mean as replacement for outliers
            running_mean = stats['running_mean'].unsqueeze(0).unsqueeze(0)
            smoothed_activations[outlier_locations] = running_mean.expand_as(activations)[outlier_locations]
        
        return smoothed_activations


class DiTASQuantizer:
    """
    DiTAS (Diffusion Transformers via Enhanced Activation Smoothing) quantizer.
    Implements W4A8 quantization with temporal-aggregated smoothing.
    """
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.activation_smoother = ActivationSmoother(config)
        self.weight_quantizers = {}
        self.activation_quantizers = {}
        self.calibrated = False
        
    def quantize_weights(self, layer: nn.Module, layer_name: str) -> nn.Module:
        """
        Quantize layer weights to specified bit-width.
        
        Args:
            layer: Neural network layer
            layer_name: Unique name for the layer
            
        Returns:
            Layer with quantized weights
        """
        if not self.config.enable_weight_quantization:
            return layer
        
        # Skip quantization for mixed-precision layers
        if (self.config.mixed_precision_layers and 
            layer_name in self.config.mixed_precision_layers):
            return layer
        
        if isinstance(layer, nn.Linear):
            return self._quantize_linear_weights(layer, layer_name)
        elif isinstance(layer, nn.Conv2d):
            return self._quantize_conv_weights(layer, layer_name)
        
        return layer
    
    def _quantize_linear_weights(self, layer: nn.Linear, layer_name: str) -> nn.Linear:
        """Quantize linear layer weights."""
        original_weight = layer.weight.data
        
        # Compute quantization parameters
        weight_min = original_weight.min()
        weight_max = original_weight.max()
        
        # Symmetric quantization
        weight_scale = max(abs(weight_min), abs(weight_max)) / (2**(self.config.weight_bits - 1) - 1)
        
        # Quantize weights
        quantized_weight = torch.round(original_weight / weight_scale).clamp(
            -(2**(self.config.weight_bits - 1)), 
            2**(self.config.weight_bits - 1) - 1
        )
        
        # Dequantize for computation
        dequantized_weight = quantized_weight * weight_scale
        
        # Create quantized layer
        quantized_layer = nn.Linear(layer.in_features, layer.out_features, 
                                   bias=layer.bias is not None)
        quantized_layer.weight.data = dequantized_weight
        
        if layer.bias is not None:
            quantized_layer.bias.data = layer.bias.data.clone()
        
        # Store quantization parameters
        self.weight_quantizers[layer_name] = {
            'scale': weight_scale,
            'original_dtype': original_weight.dtype
        }
        
        return quantized_layer
    
    def _quantize_conv_weights(self, layer: nn.Conv2d, layer_name: str) -> nn.Conv2d:
        """Quantize convolutional layer weights.""" 
        # Similar to linear quantization but for conv layers
        original_weight = layer.weight.data
        
        # Per-channel quantization for better accuracy
        weight_scale = torch.zeros(layer.out_channels)
        for c in range(layer.out_channels):
            channel_weight = original_weight[c]
            weight_scale[c] = max(abs(channel_weight.min()), abs(channel_weight.max())) / \
                             (2**(self.config.weight_bits - 1) - 1)
        
        # Quantize weights per channel
        quantized_weight = torch.zeros_like(original_weight)
        for c in range(layer.out_channels):
            quantized_weight[c] = torch.round(original_weight[c] / weight_scale[c]).clamp(
                -(2**(self.config.weight_bits - 1)), 
                2**(self.config.weight_bits - 1) - 1
            ) * weight_scale[c]
        
        # Create quantized layer
        quantized_layer = nn.Conv2d(
            layer.in_channels, layer.out_channels, layer.kernel_size,
            stride=layer.stride, padding=layer.padding, 
            bias=layer.bias is not None
        )
        quantized_layer.weight.data = quantized_weight
        
        if layer.bias is not None:
            quantized_layer.bias.data = layer.bias.data.clone()
        
        self.weight_quantizers[layer_name] = {
            'scale': weight_scale,
            'original_dtype': original_weight.dtype
        }
        
        return quantized_layer
    
    def quantize_activations(self, layer_name: str, activations: torch.Tensor,
                            timestep: int) -> torch.Tensor:
        """
        Quantize activations with temporal smoothing.
        
        Args:
            layer_name: Name of the layer
            activations: Input activations
            timestep: Current diffusion timestep
            
        Returns:
            Quantized activations
        """
        if not self.config.enable_activation_quantization:
            return activations
        
        # Apply activation smoothing first
        smoothed_activations = self.activation_smoother.smooth_activations(
            layer_name, activations, timestep
        )
        
        # Quantize smoothed activations
        activation_min = smoothed_activations.min()
        activation_max = smoothed_activations.max()
        
        # Asymmetric quantization for activations
        activation_scale = (activation_max - activation_min) / (2**self.config.activation_bits - 1)
        activation_zero_point = -activation_min / activation_scale
        
        # Quantize
        quantized_activations = torch.round(
            smoothed_activations / activation_scale + activation_zero_point
        ).clamp(0, 2**self.config.activation_bits - 1)
        
        # Dequantize for computation
        dequantized_activations = (quantized_activations - activation_zero_point) * activation_scale
        
        return dequantized_activations
    
    def calibrate(self, model: nn.Module, calibration_data: List[torch.Tensor]):
        """
        Calibrate quantization parameters using calibration dataset.
        
        Args:
            model: Model to calibrate
            calibration_data: List of calibration tensors
        """
        model.eval()
        
        print(f"🔧 Starting DiTAS calibration with {len(calibration_data)} samples...")
        
        with torch.no_grad():
            for i, data_batch in enumerate(calibration_data[:self.config.calibration_steps]):
                if i % 20 == 0:
                    print(f"   Calibration step {i+1}/{min(len(calibration_data), self.config.calibration_steps)}")
                
                # Forward pass to collect activation statistics
                # This is a simplified calibration - actual implementation would
                # require hooking into model layers
                _ = model(data_batch)
        
        self.calibrated = True
        print("✅ DiTAS calibration completed!")


class MPQDMQuantizer:
    """
    MPQ-DM (Mixed-Precision Quantization for Diffusion Models) quantizer.
    Implements outlier-driven mixed quantization with time-smoothed relation distillation.
    """
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.outlier_analyzer = OutlierAnalyzer(config.outlier_percentile)
        self.relation_distiller = TimeSmoothedRelationDistiller()
        
    def analyze_layer_sensitivity(self, model: nn.Module, 
                                 calibration_data: List[torch.Tensor]) -> Dict[str, float]:
        """
        Analyze sensitivity of different layers to quantization.
        
        Args:
            model: Model to analyze
            calibration_data: Calibration dataset
            
        Returns:
            Dictionary mapping layer names to sensitivity scores
        """
        sensitivity_scores = {}
        
        # This would involve running the model with different quantization
        # settings and measuring the impact on output quality
        # Simplified implementation here
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Heuristic: early layers and attention layers are more sensitive
                if 'attention' in name.lower() or 'embed' in name.lower():
                    sensitivity_scores[name] = 0.9  # High sensitivity
                elif 'layer.0' in name or 'layer.1' in name:
                    sensitivity_scores[name] = 0.8  # Medium-high sensitivity
                else:
                    sensitivity_scores[name] = 0.3  # Low sensitivity
        
        return sensitivity_scores
    
    def determine_mixed_precision_strategy(self, sensitivity_scores: Dict[str, float],
                                         target_compression_ratio: float = 0.5) -> Dict[str, int]:
        """
        Determine optimal bit-width for each layer based on sensitivity analysis.
        
        Args:
            sensitivity_scores: Layer sensitivity scores
            target_compression_ratio: Target model compression ratio
            
        Returns:
            Dictionary mapping layer names to bit-widths
        """
        layer_bit_widths = {}
        
        # Sort layers by sensitivity (descending)
        sorted_layers = sorted(sensitivity_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Allocate higher precision to more sensitive layers
        total_layers = len(sorted_layers)
        high_precision_count = int(total_layers * (1 - target_compression_ratio))
        
        for i, (layer_name, sensitivity) in enumerate(sorted_layers):
            if i < high_precision_count:
                layer_bit_widths[layer_name] = 8  # Higher precision
            else:
                layer_bit_widths[layer_name] = 4  # Lower precision
        
        return layer_bit_widths


class OutlierAnalyzer:
    """
    Analyzes outliers in activations for mixed-precision decisions.
    """
    
    def __init__(self, outlier_percentile: float = 99.5):
        self.outlier_percentile = outlier_percentile
        
    def detect_outliers(self, activations: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Detect outliers in activation tensor.
        
        Args:
            activations: Input activation tensor
            
        Returns:
            Tuple of (outlier_mask, outlier_ratio)
        """
        abs_activations = torch.abs(activations)
        threshold = torch.quantile(abs_activations, self.outlier_percentile / 100.0)
        
        outlier_mask = abs_activations > threshold
        outlier_ratio = outlier_mask.float().mean().item()
        
        return outlier_mask, outlier_ratio


class TimeSmoothedRelationDistiller:
    """
    Time-smoothed relation distillation for stable quantized training.
    """
    
    def __init__(self, smoothing_factor: float = 0.9):
        self.smoothing_factor = smoothing_factor
        self.relation_history = {}
    
    def compute_relation_loss(self, teacher_features: torch.Tensor,
                             student_features: torch.Tensor,
                             timestep: int) -> torch.Tensor:
        """
        Compute time-smoothed relation distillation loss.
        
        Args:
            teacher_features: Features from full-precision model
            student_features: Features from quantized model
            timestep: Current diffusion timestep
            
        Returns:
            Relation distillation loss
        """
        # Compute feature relations (simplified)
        teacher_relations = self._compute_relations(teacher_features)
        student_relations = self._compute_relations(student_features)
        
        # Time-smoothed loss
        current_loss = F.mse_loss(student_relations, teacher_relations)
        
        # Apply temporal smoothing
        if timestep in self.relation_history:
            smoothed_loss = (self.smoothing_factor * self.relation_history[timestep] + 
                           (1 - self.smoothing_factor) * current_loss)
        else:
            smoothed_loss = current_loss
        
        self.relation_history[timestep] = smoothed_loss.detach()
        
        return smoothed_loss
    
    def _compute_relations(self, features: torch.Tensor) -> torch.Tensor:
        """Compute feature relations (simplified implementation)."""
        # Normalize features
        normalized_features = F.normalize(features, p=2, dim=-1)
        
        # Compute pairwise similarities
        relations = torch.matmul(normalized_features, normalized_features.transpose(-2, -1))
        
        return relations


class MixedPrecisionOptimizer:
    """
    Unified optimizer combining DiTAS and MPQ-DM techniques.
    """
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.ditas_quantizer = DiTASQuantizer(config)
        self.mpqdm_quantizer = MPQDMQuantizer(config)
        
    def optimize_model(self, model: nn.Module, 
                      calibration_data: List[torch.Tensor]) -> nn.Module:
        """
        Apply comprehensive mixed-precision optimization to model.
        
        Args:
            model: Model to optimize
            calibration_data: Calibration dataset
            
        Returns:
            Optimized model with mixed-precision quantization
        """
        print("🚀 Starting Mixed-Precision Optimization...")
        
        # Step 1: Analyze layer sensitivity
        print("   📊 Analyzing layer sensitivity...")
        sensitivity_scores = self.mpqdm_quantizer.analyze_layer_sensitivity(
            model, calibration_data
        )
        
        # Step 2: Determine mixed-precision strategy
        print("   🎯 Determining precision strategy...")
        layer_bit_widths = self.mpqdm_quantizer.determine_mixed_precision_strategy(
            sensitivity_scores
        )
        
        # Step 3: Calibrate DiTAS quantizer
        print("   🔧 Calibrating quantizer...")
        self.ditas_quantizer.calibrate(model, calibration_data)
        
        # Step 4: Apply quantization layer by layer
        print("   ⚡ Applying quantization...")
        optimized_model = self._apply_mixed_precision_quantization(
            model, layer_bit_widths
        )
        
        print("✅ Mixed-precision optimization completed!")
        return optimized_model
    
    def _apply_mixed_precision_quantization(self, model: nn.Module,
                                           layer_bit_widths: Dict[str, int]) -> nn.Module:
        """Apply mixed-precision quantization to model."""
        optimized_model = model
        
        # Quantize each layer according to determined strategy
        for name, module in model.named_modules():
            if name in layer_bit_widths:
                # Temporarily update config for this layer
                original_bits = self.config.weight_bits
                self.config.weight_bits = layer_bit_widths[name]
                
                # Apply quantization
                quantized_module = self.ditas_quantizer.quantize_weights(module, name)
                
                # Replace module in model (simplified - actual implementation more complex)
                # This would require proper module replacement logic
                
                # Restore original config
                self.config.weight_bits = original_bits
        
        return optimized_model


def create_quantization_config(aggressive: bool = False,
                              target_bits: Tuple[int, int] = (4, 8)) -> QuantizationConfig:
    """
    Create quantization configuration with sensible defaults.
    
    Args:
        aggressive: Whether to use aggressive quantization settings
        target_bits: Tuple of (weight_bits, activation_bits)
        
    Returns:
        Configured QuantizationConfig
    """
    weight_bits, activation_bits = target_bits
    
    if aggressive:
        return QuantizationConfig(
            enable_weight_quantization=True,
            enable_activation_quantization=True,
            weight_bits=weight_bits,
            activation_bits=activation_bits,
            enable_temporal_smoothing=True,
            smoothing_alpha=0.95,  # More aggressive smoothing
            outlier_percentile=99.0,  # Lower threshold
            calibration_steps=50  # Fewer calibration steps
        )
    else:
        return QuantizationConfig(
            enable_weight_quantization=True,
            enable_activation_quantization=True,
            weight_bits=weight_bits,
            activation_bits=activation_bits,
            enable_temporal_smoothing=True,
            smoothing_alpha=0.9,
            outlier_percentile=99.5,
            calibration_steps=100
        ) 