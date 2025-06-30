"""
Dataset loading and coordination module
"""

import os
import json
import yaml
from typing import Dict, List, Optional, Any
import torch
from .loaders import VisionLoader, StructureLoader, LabelLoader
from .transforms import ImageTransforms, StructureTransforms, LayoutTransforms


class DatasetLoader:
    """
    Main dataset loader that coordinates all three loaders and applies transforms
    """
    
    def __init__(
        self, 
        dataset_root: str, 
        split: str = 'train',
        patch_size: int = 16, 
        target_size: int = 512,
        apply_transforms: bool = True,
        transform_config: Optional[Dict[str, Any]] = None
    ):
        self.dataset_root = dataset_root
        self.split = split
        self.apply_transforms = apply_transforms
        
        # Initialize component loaders
        self.vision_loader = VisionLoader(patch_size=patch_size, target_size=target_size)
        self.structure_loader = StructureLoader()
        self.label_loader = LabelLoader()
        
        # Initialize transforms if enabled
        if self.apply_transforms:
            self._init_transforms(transform_config or {})
        
        # Load dataset configuration
        self.examples = self._load_dataset_config()
    
    def _init_transforms(self, config: Dict[str, Any]):
        """Initialize preprocessing transforms"""
        # Image transforms
        image_config = config.get('image', {})
        self.image_transform = ImageTransforms(
            target_size=image_config.get('target_size', 512),
            patch_size=image_config.get('patch_size', 16),
            center_crop=image_config.get('center_crop', True),
            normalize=image_config.get('normalize', True)
        )
        
        # Structure transforms
        structure_config = config.get('structure', {})
        self.structure_transform = StructureTransforms(
            max_sequence_length=structure_config.get('max_sequence_length', 512),
            mask_probability=structure_config.get('mask_probability', 0.15)
        )
        
        # Layout transforms
        layout_config = config.get('layout', {})
        self.layout_transform = LayoutTransforms(
            max_sequence_length=layout_config.get('max_sequence_length', 256),
            label_smoothing=layout_config.get('label_smoothing', 0.1)
        )
    
    def _load_dataset_config(self) -> List[str]:
        """Load dataset configuration from YAML"""
        config_path = os.path.join(self.dataset_root, 'dataset_config.yaml')
        
        if not os.path.exists(config_path):
            # Fallback: scan directories
            split_dir = os.path.join(self.dataset_root, self.split)
            if os.path.exists(split_dir):
                return [d for d in os.listdir(split_dir) 
                       if os.path.isdir(os.path.join(split_dir, d))]
            else:
                raise FileNotFoundError(f"Dataset configuration not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config['splits'][self.split]
    
    def load_example(self, example_id: str) -> Dict[str, torch.Tensor]:
        """
        Load and process a single example
        
        Args:
            example_id: Example identifier
            
        Returns:
            Dict containing processed tensors for vision, structure, and labels
        """
        example_dir = os.path.join(self.dataset_root, self.split, example_id)
        example_json_path = os.path.join(example_dir, 'example.json')
        
        # Load example JSON
        with open(example_json_path, 'r') as f:
            example_data = json.load(f)
        
        # Process screenshot
        screenshot_path = os.path.join(example_dir, example_data['screenshot']['path'])
        
        if self.apply_transforms:
            # Use transforms for advanced processing
            from PIL import Image
            image = Image.open(screenshot_path).convert('RGB')
            vision_data = self.image_transform(image)
            vision_patches = vision_data['patches']
            patch_positions = vision_data['patch_positions']
        else:
            # Use basic loader
            vision_patches = self.vision_loader.load_and_process(
                screenshot_path,
                example_data['screenshot']['width'],
                example_data['screenshot']['height']
            )
            patch_positions = None
        
        # Process structure
        structure_tokens, hierarchy_embeddings = self.structure_loader.load_and_process(
            example_data['structure']['data']
        )
        
        if self.apply_transforms:
            # Apply structure transforms
            structure_data = self.structure_transform(
                structure_tokens, 
                hierarchy_embeddings,
                vocab_size=len(self.structure_loader.token_to_id)
            )
        else:
            structure_data = {
                'tokens': structure_tokens,
                'hierarchy_embeddings': hierarchy_embeddings
            }
        
        # Process layout labels
        label_tokens = self.label_loader.load_and_process(
            example_data['layout']['data']
        )
        
        if self.apply_transforms:
            # Apply layout transforms
            layout_data = self.layout_transform(label_tokens)
        else:
            layout_data = {'tokens': label_tokens}
        
        # Combine all data
        result = {
            'vision_patches': vision_patches,
            'example_id': example_id
        }
        
        # Add patch positions if available
        if patch_positions is not None:
            result['patch_positions'] = patch_positions
        
        # Add structure data with prefixes to avoid conflicts
        for key, value in structure_data.items():
            result[f'structure_{key}'] = value
        
        # Add layout data with prefixes
        for key, value in layout_data.items():
            result[f'layout_{key}'] = value
        
        return result
    
    def get_vocab_sizes(self) -> Dict[str, int]:
        """Get vocabulary sizes for structure and layout"""
        return {
            'structure_vocab_size': len(self.structure_loader.token_to_id),
            'layout_vocab_size': len(self.label_loader.token_to_id)
        }
    
    def get_example_info(self, idx: int) -> Dict[str, Any]:
        """Get metadata about an example without loading it"""
        example_id = self.examples[idx]
        example_dir = os.path.join(self.dataset_root, self.split, example_id)
        example_json_path = os.path.join(example_dir, 'example.json')
        
        with open(example_json_path, 'r') as f:
            example_data = json.load(f)
        
        return {
            'id': example_data['id'],
            'screenshot_width': example_data['screenshot']['width'],
            'screenshot_height': example_data['screenshot']['height'],
            'has_props': bool(example_data['layout']['data'].get('props', {}))
        }
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example_id = self.examples[idx]
        return self.load_example(example_id)


class BatchCollator:
    """
    Custom collate function for DataLoader to handle variable-length sequences
    """
    
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of examples
        
        Args:
            batch: List of example dictionaries
            
        Returns:
            Batched tensors with proper padding
        """
        # Get all keys from first example
        keys = batch[0].keys()
        
        batched = {}
        
        for key in keys:
            if key == 'example_id':
                # Keep as list for string IDs
                batched[key] = [example[key] for example in batch]
                continue
            
            # Stack tensors
            tensors = [example[key] for example in batch]
            
            if key in ['vision_patches', 'patch_positions']:
                # Vision data - can be stacked directly (same size after transforms)
                batched[key] = torch.stack(tensors)
            elif 'attention_mask' in key or 'causal_mask' in key:
                # Masks - stack directly
                batched[key] = torch.stack(tensors)
            elif 'tokens' in key or 'labels' in key:
                # Sequence data - may need padding
                batched[key] = self._pad_sequences(tensors)
            else:
                # Default: try to stack
                try:
                    batched[key] = torch.stack(tensors)
                except:
                    # If stacking fails, keep as list
                    batched[key] = tensors
        
        return batched
    
    def _pad_sequences(self, sequences: List[torch.Tensor]) -> torch.Tensor:
        """Pad sequences to same length"""
        if not sequences:
            return torch.tensor([])
        
        # All sequences should already be padded by transforms
        # Just stack them
        try:
            return torch.stack(sequences)
        except:
            # Fallback: manual padding
            max_len = max(len(seq) for seq in sequences)
            padded = []
            
            for seq in sequences:
                if len(seq) < max_len:
                    padding = torch.full((max_len - len(seq),), self.pad_token_id, dtype=seq.dtype)
                    padded_seq = torch.cat([seq, padding])
                else:
                    padded_seq = seq[:max_len]
                padded.append(padded_seq)
            
            return torch.stack(padded)
