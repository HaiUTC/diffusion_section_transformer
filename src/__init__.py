"""
Diffusion Section Transformer - Generative AI Engine for Section Layout Generation

This package implements a multimodal transformer for converting screenshots and HTML structures
into structured section layouts using a diffusion-based approach.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .data import VisionLoader, StructuralLoader, LabelLoader, MultimodalLayoutDataset
from .data import validate_dataset, FilesystemLayoutManager

__all__ = [
    "VisionLoader", 
    "StructuralLoader",
    "LabelLoader",
    "MultimodalLayoutDataset",
    "FilesystemLayoutManager",
    "validate_dataset"
] 