"""
Data Augmentation Pipelines - Step 5 Implementation

This module implements comprehensive data augmentation strategies for each training phase:
- Aggressive screenshot augmentation (50x) for Phase 1 micro-scale training
- Structure augmentation with element reordering and hierarchy modifications
- Multi-scale and multi-resolution augmentation
- Phase-specific augmentation configurations

Reference: Step 5 specifications from instruction.md
"""

import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from typing import Dict, List, Optional, Any, Tuple, Union
import random
import json
import copy
import math
import numpy as np
from dataclasses import dataclass


@dataclass 
class AggressiveAugmentationConfig:
    """Configuration for aggressive data augmentation in Phase 1."""
    
    # Screenshot augmentation parameters
    rotation_range: Tuple[float, float] = (-15, 15)
    scale_range: Tuple[float, float] = (0.8, 1.2)
    translation_range: Tuple[float, float] = (-0.1, 0.1)
    brightness_range: Tuple[float, float] = (0.7, 1.3)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    saturation_range: Tuple[float, float] = (0.9, 1.1)
    hue_range: Tuple[float, float] = (-0.05, 0.05)
    resolution_scales: List[int] = None
    
    # Structure augmentation parameters
    enable_reordering: bool = True
    class_substitution_prob: float = 0.3
    hierarchy_modification_prob: float = 0.2
    content_abstraction_prob: float = 0.4
    wrapper_injection_prob: float = 0.25
    element_dropout_prob: float = 0.1
    
    # Augmentation control
    augmentation_factor: int = 50
    preserve_semantics: bool = True
    
    def __post_init__(self):
        if self.resolution_scales is None:
            self.resolution_scales = [256, 384, 512, 768, 1024]


class ScreenshotAugmentationPipeline:
    """
    Comprehensive screenshot augmentation pipeline implementing Phase 1 aggressive
    augmentation strategy to transform 2,000 samples into 100,000+ variations.
    """
    
    def __init__(self, config: AggressiveAugmentationConfig):
        self.config = config
        self.augmentation_count = 0
        
        # Pre-defined augmentation transforms
        self.spatial_transforms = self._create_spatial_transforms()
        self.color_transforms = self._create_color_transforms()
        self.noise_transforms = self._create_noise_transforms()
        
    def __call__(self, image: Union[Image.Image, torch.Tensor], 
                 augmentation_level: str = "aggressive") -> List[torch.Tensor]:
        """
        Apply augmentation pipeline to generate multiple variations.
        
        Args:
            image: Input screenshot image
            augmentation_level: "light", "moderate", "aggressive"
            
        Returns:
            List of augmented image tensors
        """
        if isinstance(image, torch.Tensor):
            image = TF.to_pil_image(image)
        
        augmented_images = []
        num_variants = self._get_variant_count(augmentation_level)
        
        for i in range(num_variants):
            # Apply random combination of augmentations
            augmented_image = self._apply_random_augmentation(image, i)
            augmented_images.append(augmented_image)
        
        return augmented_images
    
    def _get_variant_count(self, level: str) -> int:
        """Get number of variants based on augmentation level."""
        if level == "light":
            return 5
        elif level == "moderate":
            return 15
        elif level == "aggressive":
            return self.config.augmentation_factor
        else:
            return 10
    
    def _apply_random_augmentation(self, image: Image.Image, seed: int) -> torch.Tensor:
        """Apply random combination of augmentations."""
        random.seed(seed + self.augmentation_count)
        self.augmentation_count += 1
        
        # Start with original image
        augmented = image.copy()
        
        # 1. Spatial transformations (with probability)
        if random.random() < 0.8:
            augmented = self._apply_spatial_augmentation(augmented)
        
        # 2. Color/brightness transformations
        if random.random() < 0.9:
            augmented = self._apply_color_augmentation(augmented)
        
        # 3. Resolution scaling
        if random.random() < 0.7:
            augmented = self._apply_resolution_augmentation(augmented)
        
        # 4. Noise and blur effects
        if random.random() < 0.4:
            augmented = self._apply_noise_augmentation(augmented)
        
        # 5. Layout-specific augmentations
        if random.random() < 0.5:
            augmented = self._apply_layout_augmentation(augmented)
        
        # Convert to tensor and ensure proper format
        if not isinstance(augmented, torch.Tensor):
            augmented = TF.to_tensor(augmented)
        
        return augmented
    
    def _apply_spatial_augmentation(self, image: Image.Image) -> Image.Image:
        """Apply spatial transformations."""
        # Random rotation
        angle = random.uniform(*self.config.rotation_range)
        image = TF.rotate(image, angle, fill=255)
        
        # Random scaling and translation
        scale = random.uniform(*self.config.scale_range)
        # Translation should be fraction of image size, with random direction
        max_translate = self.config.translation_range[1]  # Use max value from config
        translate_x = random.uniform(-max_translate, max_translate)
        translate_y = random.uniform(-max_translate, max_translate)
        
        # Apply affine transformation
        image = TF.affine(image, angle=0, translate=[translate_x * image.width, translate_y * image.height], 
                         scale=scale, shear=0, fill=255)
        
        return image
    
    def _apply_color_augmentation(self, image: Image.Image) -> Image.Image:
        """Apply color and brightness augmentations."""
        # Brightness adjustment
        brightness_factor = random.uniform(*self.config.brightness_range)
        image = ImageEnhance.Brightness(image).enhance(brightness_factor)
        
        # Contrast adjustment
        contrast_factor = random.uniform(*self.config.contrast_range)
        image = ImageEnhance.Contrast(image).enhance(contrast_factor)
        
        # Saturation adjustment
        saturation_factor = random.uniform(*self.config.saturation_range)
        image = ImageEnhance.Color(image).enhance(saturation_factor)
        
        # Hue adjustment (convert to tensor for hue shift)
        if random.random() < 0.5:
            tensor_img = TF.to_tensor(image)
            hue_shift = random.uniform(*self.config.hue_range)
            tensor_img = TF.adjust_hue(tensor_img, hue_shift)
            image = TF.to_pil_image(tensor_img)
        
        return image
    
    def _apply_resolution_augmentation(self, image: Image.Image) -> Image.Image:
        """Apply multi-scale resolution augmentation."""
        original_size = image.size
        
        # Random resolution from config
        target_resolution = random.choice(self.config.resolution_scales)
        
        # Resize to target resolution
        image = image.resize((target_resolution, target_resolution), Image.BICUBIC)
        
        # Randomly resize back to different resolution
        if random.random() < 0.5:
            final_resolution = random.choice(self.config.resolution_scales)
            image = image.resize((final_resolution, final_resolution), Image.BICUBIC)
        
        return image
    
    def _apply_noise_augmentation(self, image: Image.Image) -> Image.Image:
        """Apply noise and blur effects."""
        # Convert to tensor for noise operations
        tensor_img = TF.to_tensor(image)
        
        # Gaussian noise
        if random.random() < 0.3:
            noise_std = random.uniform(0.01, 0.03)
            noise = torch.randn_like(tensor_img) * noise_std
            tensor_img = torch.clamp(tensor_img + noise, 0, 1)
        
        # Convert back to PIL for blur operations
        image = TF.to_pil_image(tensor_img)
        
        # Gaussian blur
        if random.random() < 0.2:
            blur_radius = random.uniform(0.5, 1.5)
            image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        return image
    
    def _apply_layout_augmentation(self, image: Image.Image) -> Image.Image:
        """Apply layout-specific augmentations like perspective changes."""
        if random.random() < 0.3:
            # Perspective transformation
            width, height = image.size
            perspective_strength = random.uniform(0.05, 0.15)
            
            # Define perspective transformation points
            startpoints = [[0, 0], [width, 0], [width, height], [0, height]]
            endpoints = [
                [random.uniform(0, perspective_strength * width), 
                 random.uniform(0, perspective_strength * height)],
                [width - random.uniform(0, perspective_strength * width), 
                 random.uniform(0, perspective_strength * height)],
                [width - random.uniform(0, perspective_strength * width), 
                 height - random.uniform(0, perspective_strength * height)],
                [random.uniform(0, perspective_strength * width), 
                 height - random.uniform(0, perspective_strength * height)]
            ]
            
            # Apply perspective transformation
            coeffs = self._get_perspective_transform_coeffs(startpoints, endpoints)
            image = image.transform((width, height), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
        
        return image
    
    def _get_perspective_transform_coeffs(self, startpoints: List[List[float]], 
                                        endpoints: List[List[float]]) -> List[float]:
        """Calculate perspective transformation coefficients."""
        # Simplified perspective transform calculation
        matrix = []
        for i in range(4):
            matrix.append([startpoints[i][0], startpoints[i][1], 1, 0, 0, 0, 
                         -endpoints[i][0] * startpoints[i][0], -endpoints[i][0] * startpoints[i][1]])
            matrix.append([0, 0, 0, startpoints[i][0], startpoints[i][1], 1, 
                         -endpoints[i][1] * startpoints[i][0], -endpoints[i][1] * startpoints[i][1]])
        
        A = np.array(matrix, dtype=np.float32)
        B = np.array([endpoints[i][j] for i in range(4) for j in range(2)], dtype=np.float32)
        
        try:
            coeffs = np.linalg.solve(A, B)
            return coeffs.tolist()
        except np.linalg.LinAlgError:
            # Return identity transformation if solve fails
            return [1, 0, 0, 0, 1, 0, 0, 0]
    
    def _create_spatial_transforms(self) -> List[Any]:
        """Create spatial transformation functions."""
        return [
            T.RandomRotation(degrees=self.config.rotation_range),
            T.RandomAffine(degrees=0, translate=self.config.translation_range, 
                          scale=self.config.scale_range),
            T.RandomPerspective(distortion_scale=0.1, p=0.3)
        ]
    
    def _create_color_transforms(self) -> List[Any]:
        """Create color transformation functions.""" 
        return [
            T.ColorJitter(brightness=self.config.brightness_range, 
                         contrast=self.config.contrast_range,
                         saturation=self.config.saturation_range,
                         hue=self.config.hue_range)
        ]
    
    def _create_noise_transforms(self) -> List[Any]:
        """Create noise transformation functions."""
        return [
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        ]


class StructureAugmentationPipeline:
    """
    Structure augmentation pipeline for HTML object data augmentation.
    
    Implements semantic-preserving transformations:
    - Element reordering while preserving hierarchy
    - Class name variations with semantically equivalent alternatives  
    - Hierarchy modifications (wrapper injection, flattening)
    - Content abstraction with placeholder tokens
    """
    
    def __init__(self, config: AggressiveAugmentationConfig):
        self.config = config
        
        # Semantic equivalence mappings for class substitution
        self.class_substitutions = self._build_class_substitution_map()
        self.wrapper_types = ['div', 'section', 'article', 'aside', 'header', 'footer']
        self.placeholder_texts = ['[TEXT]', '[CONTENT]', '[PLACEHOLDER]', '[DATA]']
        
    def __call__(self, structure_data: Dict[str, Any], 
                 augmentation_level: str = "aggressive") -> List[Dict[str, Any]]:
        """
        Generate multiple augmented versions of HTML structure.
        
        Args:
            structure_data: Original HTML structure dictionary
            augmentation_level: Intensity of augmentation
            
        Returns:
            List of augmented structure dictionaries
        """
        augmented_structures = []
        num_variants = self._get_variant_count(augmentation_level)
        
        for i in range(num_variants):
            augmented = self._apply_structure_augmentation(structure_data, i)
            augmented_structures.append(augmented)
        
        return augmented_structures
    
    def _get_variant_count(self, level: str) -> int:
        """Get number of structure variants based on augmentation level."""
        if level == "light":
            return 3
        elif level == "moderate":
            return 8  
        elif level == "aggressive":
            return 15  # Fewer than screenshots since structure is more constrained
        else:
            return 5
    
    def _apply_structure_augmentation(self, structure_data: Dict[str, Any], 
                                    seed: int) -> Dict[str, Any]:
        """Apply random structure augmentation."""
        random.seed(seed + 1000)  # Different seed space from screenshots
        
        # Deep copy to avoid modifying original
        augmented = copy.deepcopy(structure_data)
        
        # Apply different augmentation techniques
        if self.config.enable_reordering and random.random() < 0.6:
            augmented = self._apply_element_reordering(augmented)
        
        if random.random() < self.config.class_substitution_prob:
            augmented = self._apply_class_substitution(augmented)
        
        if random.random() < self.config.hierarchy_modification_prob:
            augmented = self._apply_hierarchy_modification(augmented)
        
        if random.random() < self.config.content_abstraction_prob:
            augmented = self._apply_content_abstraction(augmented)
        
        if random.random() < self.config.wrapper_injection_prob:
            augmented = self._apply_wrapper_injection(augmented)
        
        if random.random() < self.config.element_dropout_prob:
            augmented = self._apply_element_dropout(augmented)
        
        return augmented
    
    def _apply_element_reordering(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Reorder sibling elements while preserving hierarchy."""
        def reorder_recursive(obj):
            if isinstance(obj, dict):
                # Separate text content from nested elements
                text_items = {k: v for k, v in obj.items() if k == 'text' or not isinstance(v, dict)}
                element_items = {k: v for k, v in obj.items() if k != 'text' and isinstance(v, dict)}
                
                # Reorder elements randomly
                if len(element_items) > 1:
                    element_keys = list(element_items.keys())
                    random.shuffle(element_keys)
                    element_items = {k: element_items[k] for k in element_keys}
                
                # Recursively process nested elements
                for key, value in element_items.items():
                    element_items[key] = reorder_recursive(value)
                
                # Combine text and reordered elements
                return {**text_items, **element_items}
            
            return obj
        
        return reorder_recursive(structure)
    
    def _apply_class_substitution(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Substitute class names with semantically equivalent alternatives."""
        def substitute_recursive(obj):
            if isinstance(obj, dict):
                new_obj = {}
                for key, value in obj.items():
                    # Process the key (which contains element.class format)
                    new_key = self._substitute_class_in_key(key)
                    
                    # Recursively process nested structure
                    if isinstance(value, dict):
                        new_obj[new_key] = substitute_recursive(value)
                    else:
                        new_obj[new_key] = value
                
                return new_obj
            
            return obj
        
        return substitute_recursive(structure)
    
    def _substitute_class_in_key(self, key: str) -> str:
        """Substitute class names in element keys."""
        if '.' not in key:
            return key
        
        parts = key.split('.')
        element = parts[0]
        classes = parts[1:] if len(parts) > 1 else []
        
        # Substitute classes with equivalent alternatives
        new_classes = []
        for cls in classes:
            if cls in self.class_substitutions and random.random() < 0.5:
                alternatives = self.class_substitutions[cls]
                new_cls = random.choice(alternatives)
                new_classes.append(new_cls)
            else:
                new_classes.append(cls)
        
        # Reconstruct key
        if new_classes:
            return f"{element}.{'.'.join(new_classes)}"
        else:
            return element
    
    def _apply_hierarchy_modification(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Modify hierarchy by introducing wrappers or flattening."""
        def modify_recursive(obj, depth=0):
            if isinstance(obj, dict) and depth < 4:  # Limit recursion depth
                
                # Random choice: inject wrapper or flatten
                modification_type = random.choice(['wrapper', 'flatten', 'none'])
                
                if modification_type == 'wrapper' and len(obj) > 1:
                    # Inject wrapper around some elements
                    return self._inject_wrapper(obj, depth)
                elif modification_type == 'flatten':
                    # Flatten nested structure
                    return self._flatten_structure(obj)
                else:
                    # No modification, just recurse
                    new_obj = {}
                    for key, value in obj.items():
                        if isinstance(value, dict):
                            new_obj[key] = modify_recursive(value, depth + 1)
                        else:
                            new_obj[key] = value
                    return new_obj
            
            return obj
        
        return modify_recursive(structure)
    
    def _inject_wrapper(self, structure: Dict[str, Any], depth: int) -> Dict[str, Any]:
        """Inject wrapper element around existing elements."""
        items = list(structure.items())
        
        if len(items) < 2:
            return structure
        
        # Choose random subset of elements to wrap
        num_to_wrap = random.randint(2, min(len(items), 4))
        elements_to_wrap = random.sample(items, num_to_wrap)
        remaining_elements = [item for item in items if item not in elements_to_wrap]
        
        # Create wrapper
        wrapper_type = random.choice(self.wrapper_types)
        wrapper_class = f"wrapper-{random.randint(1, 100)}"
        wrapper_key = f"{wrapper_type}.{wrapper_class}"
        
        # Build wrapped structure
        wrapped_content = {key: value for key, value in elements_to_wrap}
        
        # Combine with remaining elements
        result = {key: value for key, value in remaining_elements}
        result[wrapper_key] = wrapped_content
        
        return result
    
    def _flatten_structure(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested structure by one level."""
        flattened = {}
        
        for key, value in structure.items():
            if isinstance(value, dict) and len(value) == 1 and 'text' not in value:
                # Flatten single-child containers
                child_key, child_value = next(iter(value.items()))
                # Merge parent and child keys
                merged_key = f"{key}@{child_key}"
                flattened[merged_key] = child_value
            else:
                flattened[key] = value
        
        return flattened
    
    def _apply_content_abstraction(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Replace text content with placeholder tokens."""
        def abstract_recursive(obj):
            if isinstance(obj, dict):
                new_obj = {}
                for key, value in obj.items():
                    if key == 'text' and isinstance(value, str):
                        # Replace with placeholder
                        new_obj[key] = random.choice(self.placeholder_texts)
                    elif isinstance(value, dict):
                        new_obj[key] = abstract_recursive(value)
                    else:
                        new_obj[key] = value
                return new_obj
            
            return obj
        
        return abstract_recursive(structure)
    
    def _apply_wrapper_injection(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Inject additional wrapper elements."""
        if random.random() < 0.5:
            wrapper_type = random.choice(self.wrapper_types)
            wrapper_class = f"injected-{random.randint(1, 50)}"
            wrapper_key = f"{wrapper_type}.{wrapper_class}"
            
            return {wrapper_key: structure}
        
        return structure
    
    def _apply_element_dropout(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Randomly drop some elements (with low probability)."""
        def dropout_recursive(obj):
            if isinstance(obj, dict):
                new_obj = {}
                for key, value in obj.items():
                    # Don't drop text content or with low probability
                    if key == 'text' or random.random() > 0.1:
                        if isinstance(value, dict):
                            new_obj[key] = dropout_recursive(value)
                        else:
                            new_obj[key] = value
                return new_obj
            
            return obj
        
        return dropout_recursive(structure)
    
    def _build_class_substitution_map(self) -> Dict[str, List[str]]:
        """Build semantic equivalence mapping for class substitutions."""
        return {
            'container': ['wrapper', 'content', 'main', 'section'],
            'wrapper': ['container', 'content', 'holder'],
            'content': ['main', 'body', 'wrapper', 'container'],
            'header': ['top', 'head', 'banner'],
            'footer': ['bottom', 'foot', 'end'],
            'nav': ['navigation', 'menu', 'links'],
            'sidebar': ['aside', 'secondary', 'side'],
            'main': ['primary', 'content', 'central'],
            'grid': ['layout', 'columns', 'flex'],
            'column': ['col', 'section', 'area'],
            'padding': ['space', 'margin', 'gap'],
            'text': ['content', 'copy', 'typography'],
            'heading': ['title', 'header', 'caption'],
            'paragraph': ['text', 'copy', 'content'],
            'button': ['btn', 'action', 'control'],
            'link': ['anchor', 'url', 'href'],
            'image': ['img', 'photo', 'picture'],
            'video': ['media', 'player', 'clip']
        }


class CombinedAugmentationPipeline:
    """
    Combined augmentation pipeline that coordinates screenshot and structure
    augmentation to ensure semantic consistency between visual and structural
    modifications.
    """
    
    def __init__(self, config: AggressiveAugmentationConfig):
        self.config = config
        self.screenshot_pipeline = ScreenshotAugmentationPipeline(config)
        self.structure_pipeline = StructureAugmentationPipeline(config)
        
    def __call__(self, screenshot: Union[Image.Image, torch.Tensor],
                 structure_data: Dict[str, Any],
                 layout_data: Dict[str, Any],
                 augmentation_level: str = "aggressive") -> List[Dict[str, Any]]:
        """
        Generate coordinated augmentations of screenshot, structure, and layout.
        
        Args:
            screenshot: Input screenshot image
            structure_data: HTML structure data
            layout_data: Layout data (unchanged in augmentation)
            augmentation_level: Augmentation intensity
            
        Returns:
            List of augmented examples with consistent modifications
        """
        augmented_examples = []
        
        # Get number of augmentations to generate
        if augmentation_level == "aggressive":
            num_augmentations = self.config.augmentation_factor
        elif augmentation_level == "moderate":
            num_augmentations = 15
        else:
            num_augmentations = 5
        
        for i in range(num_augmentations):
            # Generate augmented screenshot
            augmented_screenshots = self.screenshot_pipeline(
                screenshot, augmentation_level="light"  # Generate 1 per iteration
            )
            augmented_screenshot = augmented_screenshots[0]
            
            # Generate augmented structure (use same seed for consistency)
            augmented_structures = self.structure_pipeline(
                structure_data, augmentation_level="light"
            )
            augmented_structure = augmented_structures[0]
            
            # Layout data remains unchanged (ground truth)
            augmented_example = {
                'screenshot': augmented_screenshot,
                'structure': augmented_structure,
                'layout': layout_data,  # Preserve original layout
                'augmentation_id': i
            }
            
            augmented_examples.append(augmented_example)
        
        return augmented_examples
    
    def batch_augment(self, examples: List[Dict[str, Any]], 
                     augmentation_level: str = "aggressive") -> List[Dict[str, Any]]:
        """
        Apply batch augmentation to multiple examples.
        
        Args:
            examples: List of original examples
            augmentation_level: Augmentation intensity
            
        Returns:
            Expanded list of augmented examples
        """
        all_augmented = []
        
        for idx, example in enumerate(examples):
            print(f"Augmenting example {idx + 1}/{len(examples)}...")
            
            augmented_variants = self(
                example['screenshot'],
                example['structure'], 
                example['layout'],
                augmentation_level
            )
            
            # Add original example metadata to augmented variants
            for variant in augmented_variants:
                variant['original_id'] = idx
                variant['variant_type'] = 'augmented'
            
            all_augmented.extend(augmented_variants)
        
        print(f"Generated {len(all_augmented)} augmented examples from {len(examples)} originals")
        return all_augmented


def create_augmentation_config(phase: str) -> AggressiveAugmentationConfig:
    """
    Create augmentation configuration appropriate for each training phase.
    
    Args:
        phase: Training phase ("phase1", "phase2", "phase3", "phase4")
        
    Returns:
        Configured augmentation parameters
    """
    if phase == "phase1":
        # Aggressive augmentation for micro-scale training
        return AggressiveAugmentationConfig(
            rotation_range=(-15, 15),
            scale_range=(0.8, 1.2),
            translation_range=(0.0, 0.1),  # Fixed: use positive values only
            brightness_range=(0.7, 1.3),
            contrast_range=(0.8, 1.2),
            saturation_range=(0.9, 1.1),
            resolution_scales=[256, 384, 512, 768, 1024],
            enable_reordering=True,
            class_substitution_prob=0.3,
            hierarchy_modification_prob=0.2,
            content_abstraction_prob=0.4,
            augmentation_factor=50,
            preserve_semantics=True
        )
    
    elif phase == "phase2":
        # Moderate augmentation for small-scale training  
        return AggressiveAugmentationConfig(
            rotation_range=(-10, 10),
            scale_range=(0.9, 1.1),
            translation_range=(0.0, 0.05),  # Fixed: use positive values only
            brightness_range=(0.8, 1.2),
            contrast_range=(0.9, 1.1),
            saturation_range=(0.95, 1.05),
            resolution_scales=[384, 512, 768],
            enable_reordering=True,
            class_substitution_prob=0.2,
            hierarchy_modification_prob=0.15,
            content_abstraction_prob=0.3,
            augmentation_factor=10,
            preserve_semantics=True
        )
    
    elif phase == "phase3":
        # Light augmentation for medium-scale training
        return AggressiveAugmentationConfig(
            rotation_range=(-5, 5),
            scale_range=(0.95, 1.05),
            translation_range=(0.0, 0.02),  # Fixed: use positive values only
            brightness_range=(0.9, 1.1),
            contrast_range=(0.95, 1.05),
            saturation_range=(0.98, 1.02),
            resolution_scales=[512, 768],
            enable_reordering=False,
            class_substitution_prob=0.1,
            hierarchy_modification_prob=0.05,
            content_abstraction_prob=0.1,
            augmentation_factor=5,
            preserve_semantics=True
        )
    
    elif phase == "phase4":
        # Minimal augmentation for large-scale training
        return AggressiveAugmentationConfig(
            rotation_range=(-2, 2),
            scale_range=(0.98, 1.02),
            translation_range=(0.0, 0.01),  # Fixed: use positive values only
            brightness_range=(0.95, 1.05),
            contrast_range=(0.98, 1.02),
            saturation_range=(0.99, 1.01),
            resolution_scales=[512],
            enable_reordering=False,
            class_substitution_prob=0.05,
            hierarchy_modification_prob=0.02,
            content_abstraction_prob=0.05,
            augmentation_factor=2,
            preserve_semantics=True
        )
    
    else:
        raise ValueError(f"Unknown phase: {phase}")


def demonstrate_augmentation_pipeline():
    """Demonstrate the augmentation pipeline with sample data."""
    
    # Create sample data
    sample_screenshot = Image.new('RGB', (512, 512), color=(255, 255, 255))
    sample_structure = {
        "div.container": {
            "h1.heading": {"text": "Hello World"},
            "p.paragraph": {"text": "This is a paragraph"}
        }
    }
    sample_layout = {
        "structure": {
            "section@div.container": {
                "heading@h1.heading": "",
                "paragraph@p.paragraph": ""
            }
        },
        "props": {}
    }
    
    # Create pipeline for Phase 1
    config = create_augmentation_config("phase1")
    pipeline = CombinedAugmentationPipeline(config)
    
    # Generate augmentations
    augmented_examples = pipeline(
        sample_screenshot, sample_structure, sample_layout, "moderate"
    )
    
    print(f"Generated {len(augmented_examples)} augmented examples")
    
    # Show structure variations
    for i, example in enumerate(augmented_examples[:3]):
        print(f"\nAugmented example {i + 1}:")
        print(f"Structure: {example['structure']}")
        print(f"Screenshot shape: {example['screenshot'].shape}")


if __name__ == "__main__":
    demonstrate_augmentation_pipeline() 