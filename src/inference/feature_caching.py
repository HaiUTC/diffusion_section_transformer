"""
Feature Caching and Reuse - SmoothCache Implementation

This module implements SmoothCache adaptive caching mechanism that exploits
high similarity between layer outputs across adjacent diffusion timesteps.
Achieves 8% to 71% speedup by intelligently caching and reusing features.

Reference: SmoothCache adaptive caching for diffusion transformers
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
import math
from dataclasses import dataclass, field
from collections import OrderedDict
import hashlib
import time


@dataclass 
class CachePolicy:
    """Configuration for caching behavior."""
    max_cache_size: int = 1000  # Maximum number of cached entries
    similarity_threshold: float = 0.95  # Minimum similarity for reuse
    temporal_window: int = 3  # Number of adjacent timesteps to check
    layer_specific_thresholds: Dict[str, float] = field(default_factory=dict)
    enable_compression: bool = True  # Compress cached features
    cache_ttl: float = 300.0  # Cache time-to-live in seconds


@dataclass
class CacheMetrics:
    """Metrics for cache performance monitoring."""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    total_savings_time: float = 0.0
    memory_usage_mb: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        return self.cache_hits / max(self.total_requests, 1)
    
    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate."""
        return self.cache_misses / max(self.total_requests, 1)


class TemporalSimilarityAnalyzer:
    """
    Analyzes layer-wise representation errors from calibration set to determine
    optimal caching strategies and similarity thresholds.
    """
    
    def __init__(self, similarity_threshold: float = 0.95):
        self.similarity_threshold = similarity_threshold
        self.calibration_data = {}
        self.layer_similarity_profiles = {}
        
    def analyze_temporal_similarity(self, layer_name: str, 
                                   features: torch.Tensor,
                                   timestep: int) -> float:
        """
        Analyze temporal similarity between consecutive diffusion steps.
        
        Args:
            layer_name: Name of the layer being analyzed
            features: Feature tensor from current timestep
            timestep: Current diffusion timestep
            
        Returns:
            Similarity score with previous timestep
        """
        if layer_name not in self.calibration_data:
            self.calibration_data[layer_name] = {}
        
        # Store current features
        self.calibration_data[layer_name][timestep] = features.detach().clone()
        
        # Compute similarity with adjacent timesteps
        similarities = []
        for t_offset in [-1, 1]:
            adjacent_t = timestep + t_offset
            if adjacent_t in self.calibration_data[layer_name]:
                adjacent_features = self.calibration_data[layer_name][adjacent_t]
                similarity = self._compute_feature_similarity(features, adjacent_features)
                similarities.append(similarity)
        
        # Return average similarity if available
        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            self._update_similarity_profile(layer_name, timestep, avg_similarity)
            return avg_similarity
        
        return 0.0  # No adjacent timesteps available
    
    def _compute_feature_similarity(self, features1: torch.Tensor, 
                                   features2: torch.Tensor) -> float:
        """
        Compute similarity between two feature tensors.
        
        Args:
            features1: First feature tensor
            features2: Second feature tensor
            
        Returns:
            Similarity score between 0 and 1
        """
        if features1.shape != features2.shape:
            return 0.0
        
        # Flatten tensors for similarity computation
        flat1 = features1.view(-1)
        flat2 = features2.view(-1)
        
        # Cosine similarity
        cosine_sim = torch.cosine_similarity(flat1.unsqueeze(0), flat2.unsqueeze(0))
        
        # L2 similarity (normalized)
        l2_distance = torch.norm(flat1 - flat2)
        l2_sim = 1.0 / (1.0 + l2_distance.item())
        
        # Combined similarity score
        combined_sim = 0.7 * cosine_sim.item() + 0.3 * l2_sim
        
        return max(0.0, min(1.0, combined_sim))
    
    def _update_similarity_profile(self, layer_name: str, timestep: int, similarity: float):
        """Update similarity profile for a layer."""
        if layer_name not in self.layer_similarity_profiles:
            self.layer_similarity_profiles[layer_name] = {}
        
        self.layer_similarity_profiles[layer_name][timestep] = similarity
    
    def get_adaptive_threshold(self, layer_name: str, timestep: int) -> float:
        """
        Get adaptive similarity threshold based on layer and timestep characteristics.
        
        Args:
            layer_name: Name of the layer
            timestep: Current timestep
            
        Returns:
            Adaptive similarity threshold
        """
        base_threshold = self.similarity_threshold
        
        # Adjust threshold based on layer similarity profile
        if (layer_name in self.layer_similarity_profiles and 
            timestep in self.layer_similarity_profiles[layer_name]):
            
            historical_similarity = self.layer_similarity_profiles[layer_name][timestep]
            
            # Lower threshold for layers with historically high similarity
            if historical_similarity > 0.9:
                return base_threshold * 0.9
            elif historical_similarity < 0.7:
                return base_threshold * 1.1
        
        return base_threshold
    
    def cleanup_old_data(self, max_age_steps: int = 100):
        """Clean up old calibration data to prevent memory bloat."""
        for layer_name in list(self.calibration_data.keys()):
            layer_data = self.calibration_data[layer_name]
            
            # Keep only recent timesteps
            recent_timesteps = sorted(layer_data.keys())[-max_age_steps:]
            
            self.calibration_data[layer_name] = {
                t: layer_data[t] for t in recent_timesteps
            }


class SmoothCache:
    """
    SmoothCache implementation with adaptive caching mechanism.
    Exploits temporal redundancy between adjacent diffusion steps to reuse features.
    """
    
    def __init__(self, policy: CachePolicy):
        self.policy = policy
        self.cache = OrderedDict()  # LRU cache implementation
        self.metrics = CacheMetrics()
        self.similarity_analyzer = TemporalSimilarityAnalyzer(policy.similarity_threshold)
        
    def _generate_cache_key(self, layer_name: str, timestep: int, 
                           input_hash: str) -> str:
        """Generate unique cache key for layer output."""
        return f"{layer_name}_t{timestep}_{input_hash[:8]}"
    
    def _hash_tensor(self, tensor: torch.Tensor) -> str:
        """Generate hash for input tensor."""
        # Use tensor shape and sample of values for efficient hashing
        tensor_bytes = tensor.detach().cpu().numpy().tobytes()
        return hashlib.md5(tensor_bytes).hexdigest()
    
    def _compress_features(self, features: torch.Tensor) -> torch.Tensor:
        """Compress features if compression is enabled."""
        if not self.policy.enable_compression:
            return features
        
        # Simple compression using half precision
        return features.half()
    
    def _decompress_features(self, compressed_features: torch.Tensor, 
                            target_dtype: torch.dtype) -> torch.Tensor:
        """Decompress features to target dtype."""
        if not self.policy.enable_compression:
            return compressed_features
        
        return compressed_features.to(target_dtype)
    
    def get_cached_features(self, layer_name: str, timestep: int,
                           input_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Retrieve cached features if similarity threshold is met.
        
        Args:
            layer_name: Name of the layer
            timestep: Current diffusion timestep
            input_tensor: Input tensor for similarity checking
            
        Returns:
            Cached features if available and similar enough, None otherwise
        """
        self.metrics.total_requests += 1
        
        input_hash = self._hash_tensor(input_tensor)
        
        # Check for exact match first
        exact_key = self._generate_cache_key(layer_name, timestep, input_hash)
        if exact_key in self.cache:
            entry = self.cache[exact_key]
            if not self._is_expired(entry):
                # Move to end (LRU)
                self.cache.move_to_end(exact_key)
                self.metrics.cache_hits += 1
                cached_features = self._decompress_features(
                    entry['features'], input_tensor.dtype
                )
                return cached_features
        
        # Check temporal window for similar features
        adaptive_threshold = self.similarity_analyzer.get_adaptive_threshold(
            layer_name, timestep
        )
        
        for t_offset in range(-self.policy.temporal_window, 
                             self.policy.temporal_window + 1):
            if t_offset == 0:
                continue  # Skip current timestep
                
            candidate_timestep = timestep + t_offset
            candidate_key = self._generate_cache_key(layer_name, candidate_timestep, input_hash)
            
            if candidate_key in self.cache:
                entry = self.cache[candidate_key]
                if not self._is_expired(entry):
                    # Check similarity
                    similarity = self.similarity_analyzer.analyze_temporal_similarity(
                        layer_name, input_tensor, timestep
                    )
                    
                    if similarity >= adaptive_threshold:
                        # Cache hit with temporal similarity
                        self.cache.move_to_end(candidate_key)
                        self.metrics.cache_hits += 1
                        cached_features = self._decompress_features(
                            entry['features'], input_tensor.dtype
                        )
                        return cached_features
        
        # Cache miss
        self.metrics.cache_misses += 1
        return None
    
    def cache_features(self, layer_name: str, timestep: int,
                      input_tensor: torch.Tensor, 
                      output_features: torch.Tensor) -> None:
        """
        Cache computed features for future reuse.
        
        Args:
            layer_name: Name of the layer
            timestep: Current diffusion timestep
            input_tensor: Input tensor (for hashing)
            output_features: Computed output features to cache
        """
        # Check cache size limit
        if len(self.cache) >= self.policy.max_cache_size:
            self._evict_oldest()
        
        input_hash = self._hash_tensor(input_tensor)
        cache_key = self._generate_cache_key(layer_name, timestep, input_hash)
        
        # Store compressed features
        compressed_features = self._compress_features(output_features.detach().clone())
        
        cache_entry = {
            'features': compressed_features,
            'timestamp': time.time(),
            'layer_name': layer_name,
            'timestep': timestep,
            'memory_size': compressed_features.numel() * compressed_features.element_size()
        }
        
        self.cache[cache_key] = cache_entry
        
        # Update memory usage
        self._update_memory_usage()
    
    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if cache entry has expired."""
        return (time.time() - entry['timestamp']) > self.policy.cache_ttl
    
    def _evict_oldest(self) -> None:
        """Evict oldest cache entry (LRU)."""
        if self.cache:
            self.cache.popitem(last=False)
            self.metrics.evictions += 1
    
    def _update_memory_usage(self) -> None:
        """Update memory usage metrics."""
        total_memory = sum(entry['memory_size'] for entry in self.cache.values())
        self.metrics.memory_usage_mb = total_memory / (1024 * 1024)
    
    def clear_cache(self) -> None:
        """Clear all cached features."""
        self.cache.clear()
        self.metrics = CacheMetrics()
    
    def get_metrics(self) -> CacheMetrics:
        """Get current cache performance metrics."""
        return self.metrics


class FeatureCacheManager:
    """
    High-level manager for feature caching across multiple layers and models.
    Coordinates caching policies and provides unified interface.
    """
    
    def __init__(self, global_policy: CachePolicy):
        self.global_policy = global_policy
        self.layer_caches: Dict[str, SmoothCache] = {}
        self.global_metrics = CacheMetrics()
        
    def get_or_create_cache(self, layer_name: str) -> SmoothCache:
        """Get existing cache or create new one for layer."""
        if layer_name not in self.layer_caches:
            # Create layer-specific policy if configured
            layer_policy = self._get_layer_policy(layer_name)
            self.layer_caches[layer_name] = SmoothCache(layer_policy)
        
        return self.layer_caches[layer_name]
    
    def _get_layer_policy(self, layer_name: str) -> CachePolicy:
        """Get layer-specific caching policy."""
        policy = CachePolicy(
            max_cache_size=self.global_policy.max_cache_size,
            similarity_threshold=self.global_policy.similarity_threshold,
            temporal_window=self.global_policy.temporal_window,
            enable_compression=self.global_policy.enable_compression,
            cache_ttl=self.global_policy.cache_ttl
        )
        
        # Apply layer-specific threshold if configured
        if layer_name in self.global_policy.layer_specific_thresholds:
            policy.similarity_threshold = self.global_policy.layer_specific_thresholds[layer_name]
        
        return policy
    
    def cached_forward(self, layer: nn.Module, layer_name: str,
                      input_tensor: torch.Tensor, timestep: int,
                      *args, **kwargs) -> torch.Tensor:
        """
        Execute layer forward pass with caching.
        
        Args:
            layer: Neural network layer
            layer_name: Unique name for the layer
            input_tensor: Input tensor
            timestep: Current diffusion timestep
            *args, **kwargs: Additional arguments for layer forward
            
        Returns:
            Layer output (cached or computed)
        """
        cache = self.get_or_create_cache(layer_name)
        
        # Try to get cached features
        start_time = time.time()
        cached_output = cache.get_cached_features(layer_name, timestep, input_tensor)
        
        if cached_output is not None:
            # Cache hit - return cached features
            self.global_metrics.cache_hits += 1
            self.global_metrics.total_savings_time += time.time() - start_time
            return cached_output
        
        # Cache miss - compute features
        self.global_metrics.cache_misses += 1
        
        # Compute layer output
        compute_start = time.time()
        output = layer(input_tensor, *args, **kwargs)
        compute_time = time.time() - compute_start
        
        # Cache the computed output
        cache.cache_features(layer_name, timestep, input_tensor, output)
        
        return output
    
    def get_aggregated_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics across all layer caches."""
        total_requests = sum(cache.metrics.total_requests for cache in self.layer_caches.values())
        total_hits = sum(cache.metrics.cache_hits for cache in self.layer_caches.values())
        total_misses = sum(cache.metrics.cache_misses for cache in self.layer_caches.values())
        total_memory = sum(cache.metrics.memory_usage_mb for cache in self.layer_caches.values())
        
        return {
            'total_requests': total_requests,
            'total_hits': total_hits,
            'total_misses': total_misses,
            'overall_hit_rate': total_hits / max(total_requests, 1),
            'total_memory_mb': total_memory,
            'num_layer_caches': len(self.layer_caches),
            'layer_metrics': {
                name: cache.get_metrics() for name, cache in self.layer_caches.items()
            }
        }
    
    def clear_all_caches(self) -> None:
        """Clear all layer caches."""
        for cache in self.layer_caches.values():
            cache.clear_cache()
        self.global_metrics = CacheMetrics()
    
    def optimize_cache_settings(self) -> Dict[str, Any]:
        """
        Analyze cache performance and suggest optimizations.
        
        Returns:
            Dictionary with optimization suggestions
        """
        metrics = self.get_aggregated_metrics()
        suggestions = []
        
        # Analyze hit rates
        if metrics['overall_hit_rate'] < 0.3:
            suggestions.append("Consider increasing similarity thresholds")
        elif metrics['overall_hit_rate'] > 0.8:
            suggestions.append("Consider decreasing similarity thresholds for more aggressive caching")
        
        # Analyze memory usage
        if metrics['total_memory_mb'] > 1000:  # Over 1GB
            suggestions.append("Consider enabling compression or reducing cache size")
        
        # Layer-specific analysis
        layer_suggestions = {}
        for layer_name, cache in self.layer_caches.items():
            layer_metrics = cache.get_metrics()
            if layer_metrics.hit_rate < 0.2:
                layer_suggestions[layer_name] = "Low hit rate - consider layer-specific tuning"
        
        return {
            'overall_metrics': metrics,
            'suggestions': suggestions,
            'layer_suggestions': layer_suggestions
        }


def create_cache_policy(conservative: bool = False) -> CachePolicy:
    """
    Create cache policy with sensible defaults.
    
    Args:
        conservative: Whether to use conservative settings
        
    Returns:
        Configured CachePolicy
    """
    if conservative:
        return CachePolicy(
            max_cache_size=500,
            similarity_threshold=0.98,  # Very high threshold
            temporal_window=2,
            enable_compression=True,
            cache_ttl=180.0  # 3 minutes
        )
    else:
        return CachePolicy(
            max_cache_size=1000,
            similarity_threshold=0.95,  # Standard threshold
            temporal_window=3,
            enable_compression=True,
            cache_ttl=300.0  # 5 minutes
        ) 