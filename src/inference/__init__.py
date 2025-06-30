"""
Inference Pipeline & Optimization Techniques - Step 4 Implementation

This package implements comprehensive inference optimization following the specifications
in Step 4 of the instruction.md. It includes:

1. Parallel Inference Engine (xDiT-style parallelism)
2. Dynamic Execution Optimization (DyDiT techniques)
3. Feature Caching and Reuse (SmoothCache)
4. Quantization and Compression (DiTAS, MPQ-DM)
5. Real-Time Streaming Pipeline (StreamDiffusion)
6. Production Deployment Architecture
"""

# Core inference engine
from .parallel_engine import (
    ParallelInferenceEngine, SequenceParallelism, PipeFusion, 
    CFGParallel, DataParallel, HybridParallelismFramework,
    ParallelConfig, create_parallel_config
)

# Dynamic execution optimization
from .dynamic_optimization import (
    DynamicExecutionOptimizer, TimestepDynamicWidth, SpatialDynamicToken,
    AdaptiveComputationStrategy, DynamicConfig, create_dynamic_config,
    ComplexityAnalyzer
)

# Feature caching system
from .feature_caching import (
    SmoothCache, FeatureCacheManager, TemporalSimilarityAnalyzer,
    CachePolicy, CacheMetrics, create_cache_policy
)

# Quantization and compression
from .quantization import (
    DiTASQuantizer, MPQDMQuantizer, MixedPrecisionOptimizer,
    QuantizationConfig, ActivationSmoother, create_quantization_config,
    OutlierAnalyzer, TimeSmoothedRelationDistiller
)

# Production inference pipeline
from .production_pipeline import (
    ProductionInferencePipeline, RequestPreprocessor, LayoutGenerator,
    PostProcessor, QualityValidator, InferenceRequest, InferenceResponse,
    PipelineConfig, create_production_pipeline
)

__all__ = [
    # Parallel Engine
    "ParallelInferenceEngine",
    "SequenceParallelism", 
    "PipeFusion",
    "CFGParallel",
    "DataParallel",
    "HybridParallelismFramework",
    "ParallelConfig",
    "create_parallel_config",
    
    # Dynamic Optimization
    "DynamicExecutionOptimizer",
    "TimestepDynamicWidth",
    "SpatialDynamicToken", 
    "AdaptiveComputationStrategy",
    "DynamicConfig",
    "create_dynamic_config",
    "ComplexityAnalyzer",
    
    # Feature Caching
    "SmoothCache",
    "FeatureCacheManager",
    "TemporalSimilarityAnalyzer",
    "CachePolicy",
    "CacheMetrics",
    "create_cache_policy",
    
    # Quantization
    "DiTASQuantizer",
    "MPQDMQuantizer",
    "MixedPrecisionOptimizer",
    "QuantizationConfig",
    "ActivationSmoother",
    "create_quantization_config",
    "OutlierAnalyzer",
    "TimeSmoothedRelationDistiller",
    
    # Production Pipeline
    "ProductionInferencePipeline",
    "RequestPreprocessor",
    "LayoutGenerator", 
    "PostProcessor",
    "QualityValidator",
    "InferenceRequest",
    "InferenceResponse", 
    "PipelineConfig",
    "create_production_pipeline"
] 