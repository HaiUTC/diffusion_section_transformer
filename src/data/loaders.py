"""
Individual data loaders for vision, structure, and layout data processing
"""

import json
import os
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms


class VisionLoader:
    """
    Vision Loader for processing screenshot images
    - Input: screenshot.png, (width, height)
    - Process: Load, resize, normalize, tokenize into patches (ViT-style)
    """
    
    def __init__(self, patch_size: int = 16, target_size: int = 512, normalize: bool = True):
        self.patch_size = patch_size
        self.target_size = target_size
        self.normalize = normalize
        
        # Standard ImageNet normalization
        self.transform = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
        ])
        
        if normalize:
            self.transform = transforms.Compose([
                self.transform,
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def load_and_process(self, image_path: str, width: int = None, height: int = None) -> torch.Tensor:
        """
        Load and process screenshot image into patch embeddings
        
        Args:
            image_path: Path to screenshot.png
            width, height: Original image dimensions (for metadata)
            
        Returns:
            torch.Tensor: Patch embeddings of shape (num_patches, patch_dim)
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms (resize, normalize, convert to tensor)
            image_tensor = self.transform(image)
            
            # Tokenize into patches (ViT-style)
            patch_embeddings = self._tokenize_to_patches(image_tensor)
            
            return patch_embeddings
            
        except Exception as e:
            raise ValueError(f"Failed to process image {image_path}: {str(e)}")
    
    def _tokenize_to_patches(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert image tensor to non-overlapping patches
        
        Args:
            image_tensor: Tensor of shape (C, H, W)
            
        Returns:
            torch.Tensor: Patch embeddings of shape (num_patches, patch_dim)
        """
        C, H, W = image_tensor.shape
        
        # Ensure image dimensions are divisible by patch_size
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"Image size ({H}, {W}) not divisible by patch size ({self.patch_size})"
        
        # Reshape to patches
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        
        # Rearrange to patches: (C, H, W) -> (num_patches, patch_dim)
        patches = image_tensor.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(C, num_patches_h, num_patches_w, self.patch_size, self.patch_size)
        patches = patches.permute(1, 2, 0, 3, 4).contiguous()
        patches = patches.view(num_patches_h * num_patches_w, C * self.patch_size * self.patch_size)
        
        return patches


class StructureLoader:
    """
    Structure Loader for processing HTML structure JSON
    - Input: structure.data JSON
    - Process: Traverse nested object, emit tokens with hierarchy info
    """
    
    def __init__(self, vocab_file: Optional[str] = None):
        # Define special tokens first
        self.special_tokens = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[SEP]': 2,  # Separator for hierarchy levels
            '[START]': 3,
            '[END]': 4
        }
        
        # Then build vocabulary
        self.vocab = self._build_vocabulary(vocab_file)
        self.token_to_id = {**self.special_tokens, **self.vocab}
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
    
    def _build_vocabulary(self, vocab_file: Optional[str] = None) -> Dict[str, int]:
        """Build vocabulary from training data or load from file"""
        if vocab_file and os.path.exists(vocab_file):
            with open(vocab_file, 'r') as f:
                vocab = json.load(f)
            return vocab
        
        # Default vocabulary with common HTML elements and classes
        base_vocab = [
            'div', 'span', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            'p', 'a', 'img', 'button', 'header', 'footer', 'nav',
            'section', 'article', 'aside', 'main', 'ul', 'ol', 'li',
            'container', 'wrapper', 'content', 'grid', 'column',
            'padding', 'margin', 'text', 'heading', 'paragraph',
            'w-button', 'w-layout-grid', 'is-secondary'
        ]
        
        vocab = {}
        start_idx = len(self.special_tokens)
        for i, token in enumerate(base_vocab):
            vocab[token] = start_idx + i
            
        return vocab
    
    def load_and_process(self, structure_data: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process HTML structure into token sequences with hierarchy embeddings
        
        Args:
            structure_data: Nested structure dictionary
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (token_sequence, hierarchy_embeddings)
        """
        tokens = []
        hierarchy_info = []
        
        # Add start token
        tokens.append(self.token_to_id['[START]'])
        hierarchy_info.append((0, 0))  # (depth, sibling_index)
        
        # Traverse structure in preorder
        self._traverse_structure(structure_data, tokens, hierarchy_info, depth=0, sibling_idx=0)
        
        # Add end token
        tokens.append(self.token_to_id['[END]'])
        hierarchy_info.append((0, 0))
        
        # Convert to tensors
        token_tensor = torch.tensor(tokens, dtype=torch.long)
        hierarchy_tensor = torch.tensor(hierarchy_info, dtype=torch.long)
        
        return token_tensor, hierarchy_tensor
    
    def _traverse_structure(self, structure: Dict[str, Any], tokens: List[int], 
                          hierarchy_info: List[Tuple[int, int]], depth: int, sibling_idx: int):
        """Recursively traverse nested structure and emit tokens"""
        
        for i, (key, value) in enumerate(structure.items()):
            # Handle compound keys with @ concatenation
            if '@' in key:
                # Keep as single compound token
                token_id = self._get_token_id(key)
            else:
                # Parse element.class format
                token_id = self._parse_element_key(key)
            
            tokens.append(token_id)
            hierarchy_info.append((depth, i))
            
            # Add separator for hierarchy level
            tokens.append(self.token_to_id['[SEP]'])
            hierarchy_info.append((depth, i))
            
            # Handle text content
            if isinstance(value, dict):
                if 'text' in value:
                    # Add text indicator (simplified - in practice might tokenize text)
                    text_token_id = self._get_token_id('text')
                    tokens.append(text_token_id)
                    hierarchy_info.append((depth + 1, 0))
                
                # Recursively process nested elements
                nested_structure = {k: v for k, v in value.items() if k != 'text'}
                if nested_structure:
                    self._traverse_structure(nested_structure, tokens, hierarchy_info, depth + 1, 0)
    
    def _parse_element_key(self, key: str) -> int:
        """Parse element.class format and return token ID"""
        # Handle complex keys like "div.w-layout-grid.header1_content['d:grid']['col:2']"
        # Simplified parsing - extract main element and first class
        parts = key.split('.')
        element = parts[0] if parts else key
        
        # Try to get element token, fallback to full key
        if element in self.token_to_id:
            return self.token_to_id[element]
        else:
            return self._get_token_id(key)
    
    def _get_token_id(self, token: str) -> int:
        """Get token ID, adding to vocabulary if new"""
        if token in self.token_to_id:
            return self.token_to_id[token]
        
        # Add new token to vocabulary
        new_id = len(self.token_to_id)
        self.token_to_id[token] = new_id
        self.id_to_token[new_id] = token
        return new_id


class LabelLoader:
    """
    Label Loader for processing layout JSON into target sequences
    - Input: layout.data JSON
    - Process: Extract structure and props, create target token sequence
    """
    
    def __init__(self, vocab_file: Optional[str] = None):
        # Define special tokens first
        self.special_tokens = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[START]': 2,
            '[END]': 3,
            '[PROP_START]': 4,
            '[PROP_END]': 5
        }
        
        # Then build vocabulary
        self.vocab = self._build_layout_vocabulary(vocab_file)
        self.token_to_id = {**self.special_tokens, **self.vocab}
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
    
    def _build_layout_vocabulary(self, vocab_file: Optional[str] = None) -> Dict[str, int]:
        """Build vocabulary for layout elements and operations"""
        if vocab_file and os.path.exists(vocab_file):
            with open(vocab_file, 'r') as f:
                vocab = json.load(f)
            return vocab
        
        # Layout element vocabulary
        layout_elements = [
            'section', 'grid', 'column', 'wrapper', 'freedom',  # Layout elements
            'heading', 'paragraph', 'button', 'icon', 'image', 'list', 'map', 'qr', 
            'counter', 'divider', 'video', 'marquee',  # Basic elements
            'carousel', 'accordion', 'tab', 'social', 'gallery', 'masonry',  # Advanced elements
            'column$1', 'column$2', 'column$3', 'column$4',  # Column indices
            'bi', 'bv', 'bo',  # Props: background image, video, overlay
            '@', ':', '$'  # Special syntax characters
        ]
        
        vocab = {}
        start_idx = len(self.special_tokens)
        for i, token in enumerate(layout_elements):
            vocab[token] = start_idx + i
            
        return vocab
    
    def load_and_process(self, layout_data: Dict[str, Any]) -> torch.Tensor:
        """
        Process layout data into target token sequence
        
        Args:
            layout_data: Layout dictionary with structure and props
            
        Returns:
            torch.Tensor: Target token sequence for training
        """
        tokens = []
        
        # Add start token
        tokens.append(self.token_to_id['[START]'])
        
        # Process structure
        structure = layout_data.get('structure', {})
        self._process_layout_structure(structure, tokens)
        
        # Process props
        props = layout_data.get('props', {})
        if props:
            tokens.append(self.token_to_id['[PROP_START]'])
            self._process_layout_props(props, tokens)
            tokens.append(self.token_to_id['[PROP_END]'])
        
        # Add end token
        tokens.append(self.token_to_id['[END]'])
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def _process_layout_structure(self, structure: Dict[str, Any], tokens: List[int]):
        """Process layout structure recursively"""
        for key, value in structure.items():
            # Tokenize compound key (e.g., "section@header.section_header1@div.padding-global")
            self._tokenize_compound_key(key, tokens)
            
            if isinstance(value, dict) and value:
                # Recursively process nested structure
                self._process_layout_structure(value, tokens)
    
    def _process_layout_props(self, props: Dict[str, Any], tokens: List[int]):
        """Process layout props (background images, videos, etc.)"""
        for prop_key, prop_value in props.items():
            # Add prop key (e.g., 'bi', 'bv', 'bo')
            prop_token_id = self._get_token_id(prop_key)
            tokens.append(prop_token_id)
            
            # Add colon separator
            colon_token_id = self._get_token_id(':')
            tokens.append(colon_token_id)
            
            # Add prop value
            value_token_id = self._get_token_id(prop_value)
            tokens.append(value_token_id)
    
    def _tokenize_compound_key(self, compound_key: str, tokens: List[int]):
        """Tokenize compound key with @ separators"""
        parts = compound_key.split('@')
        
        for i, part in enumerate(parts):
            if i > 0:
                # Add @ separator token
                at_token_id = self._get_token_id('@')
                tokens.append(at_token_id)
            
            # Parse element type and class
            if '$' in part:
                # Handle column indices (e.g., "column$1")
                token_id = self._get_token_id(part)
            else:
                # Extract element type (before @ concatenation)
                element_type = part.split('.')[0] if '.' in part else part
                element_type = element_type.split('@')[0] if '@' in element_type else element_type
                token_id = self._get_token_id(element_type)
            
            tokens.append(token_id)
    
    def _get_token_id(self, token: str) -> int:
        """Get token ID, adding to vocabulary if new"""
        if token in self.token_to_id:
            return self.token_to_id[token]
        
        # Add new token to vocabulary
        new_id = len(self.token_to_id)
        self.token_to_id[token] = new_id
        self.id_to_token[new_id] = token
        return new_id 