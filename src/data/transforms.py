"""
Preprocessing transforms for Section Layout Generation - Task 2.4

This module implements the preprocessing transforms as specified in the instruction:
- Image Transforms: resize, normalize, patch embedding  
- Structure Transforms: token mapping, position embeddings, masking
- Layout Transforms: tokenization, attention masks, label smoothing
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any, Union
import random


class ImageTransforms:
    """
    Image preprocessing transforms following instruction 2.4:
    - Resize → Fixed resolution
    - Center-crop or pad to square if necessary
    - Normalize (mean/std)
    - Patch embedding (e.g., 16×16 pixels per patch)
    """
    
    def __init__(
        self,
        target_size: int = 512,
        patch_size: int = 16,
        normalize: bool = True,
        center_crop: bool = True,
        padding_mode: str = 'constant',
        padding_value: float = 0.0
    ):
        self.target_size = target_size
        self.patch_size = patch_size
        self.normalize = normalize
        self.center_crop = center_crop
        self.padding_mode = padding_mode
        self.padding_value = padding_value
        
        # Standard ImageNet normalization values
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
    
    def __call__(self, image: Union[Image.Image, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply image transforms"""
        # Convert to PIL if tensor
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        
        # Step 1: Resize and crop/pad to square
        image_tensor = self._resize_and_square(image)
        
        # Step 2: Normalize
        if self.normalize:
            image_tensor = self._normalize(image_tensor)
        
        # Step 3: Create patch embeddings
        patches, patch_positions = self._create_patches(image_tensor)
        
        return {
            'image_tensor': image_tensor,
            'patches': patches,
            'patch_positions': patch_positions
        }
    
    def _resize_and_square(self, image: Image.Image) -> torch.Tensor:
        """Resize and make image square using crop or padding"""
        to_tensor = transforms.ToTensor()
        image_tensor = to_tensor(image)
        C, H, W = image_tensor.shape
        
        if self.center_crop:
            # Center crop to square, then resize
            min_dim = min(H, W)
            crop_h = (H - min_dim) // 2
            crop_w = (W - min_dim) // 2
            image_tensor = image_tensor[:, crop_h:crop_h + min_dim, crop_w:crop_w + min_dim]
        else:
            # Pad to square
            max_dim = max(H, W)
            pad_h = (max_dim - H) // 2
            pad_w = (max_dim - W) // 2
            image_tensor = F.pad(
                image_tensor, 
                (pad_w, max_dim - W - pad_w, pad_h, max_dim - H - pad_h),
                mode=self.padding_mode,
                value=self.padding_value
            )
        
        # Resize to target size
        resize_transform = transforms.Resize((self.target_size, self.target_size))
        image_tensor = resize_transform(image_tensor)
        
        return image_tensor
    
    def _normalize(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Apply normalization"""
        normalize_transform = transforms.Normalize(mean=self.mean, std=self.std)
        return normalize_transform(image_tensor)
    
    def _create_patches(self, image_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create patch embeddings and position encodings"""
        C, H, W = image_tensor.shape
        
        # Ensure divisible by patch size
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        
        # Calculate number of patches
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        
        # Create patches using unfold
        patches = image_tensor.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(C, num_patches_h, num_patches_w, self.patch_size, self.patch_size)
        patches = patches.permute(1, 2, 0, 3, 4).contiguous()
        patches = patches.view(num_patches_h * num_patches_w, C * self.patch_size * self.patch_size)
        
        # Create position encodings
        patch_positions = torch.zeros(num_patches_h * num_patches_w, 2)
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                idx = i * num_patches_w + j
                patch_positions[idx, 0] = i / num_patches_h
                patch_positions[idx, 1] = j / num_patches_w
        
        return patches, patch_positions


class StructureTransforms:
    """
    Structure preprocessing transforms following instruction 2.4:
    - JSON → Token index mapping (vocabulary includes compound keys)
    - Position-in-tree embeddings (depth, sibling index)
    - Masking strategy for optional structure tokens (for diffusion noise injection)
    """
    
    def __init__(
        self,
        max_sequence_length: int = 512,
        mask_probability: float = 0.15,
        pad_token_id: int = 0,
        mask_token_id: int = 1
    ):
        self.max_sequence_length = max_sequence_length
        self.mask_probability = mask_probability
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
    
    def __call__(
        self, 
        tokens: torch.Tensor, 
        hierarchy_info: torch.Tensor,
        vocab_size: int = None
    ) -> Dict[str, torch.Tensor]:
        """Apply structure transforms"""
        # Step 1: Truncate or pad to max length
        tokens_padded, attention_mask = self._pad_or_truncate(tokens)
        hierarchy_padded = self._pad_hierarchy(hierarchy_info, len(tokens_padded))
        
        # Step 2: Create position embeddings
        position_embeddings = self._create_position_embeddings(hierarchy_padded)
        
        # Step 3: Create masked version for diffusion training
        masked_tokens, mask_labels = self._create_masked_tokens(tokens_padded, attention_mask, vocab_size)
        
        return {
            'tokens': tokens_padded,
            'attention_mask': attention_mask,
            'position_embeddings': position_embeddings,
            'masked_tokens': masked_tokens,
            'mask_labels': mask_labels
        }
    
    def _pad_or_truncate(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad or truncate tokens to max length"""
        seq_len = len(tokens)
        
        if seq_len >= self.max_sequence_length:
            tokens_padded = tokens[:self.max_sequence_length]
            attention_mask = torch.ones(self.max_sequence_length, dtype=torch.bool)
        else:
            padding_length = self.max_sequence_length - seq_len
            tokens_padded = F.pad(tokens, (0, padding_length), value=self.pad_token_id)
            attention_mask = torch.cat([
                torch.ones(seq_len, dtype=torch.bool),
                torch.zeros(padding_length, dtype=torch.bool)
            ])
        
        return tokens_padded, attention_mask
    
    def _pad_hierarchy(self, hierarchy_info: torch.Tensor, target_length: int) -> torch.Tensor:
        """Pad hierarchy info to target length"""
        current_length = len(hierarchy_info)
        
        if current_length >= target_length:
            return hierarchy_info[:target_length]
        else:
            padding_length = target_length - current_length
            padding = torch.zeros(padding_length, 2, dtype=hierarchy_info.dtype)
            return torch.cat([hierarchy_info, padding])
    
    def _create_position_embeddings(self, hierarchy_info: torch.Tensor) -> torch.Tensor:
        """Create position embeddings from tree structure"""
        seq_len, _ = hierarchy_info.shape
        
        # Create depth embeddings (normalized)
        depths = hierarchy_info[:, 0].float()
        max_depth = depths.max().item() if depths.max().item() > 0 else 1
        depth_embeddings = depths / max_depth
        
        # Create sibling position embeddings
        sibling_positions = hierarchy_info[:, 1].float()
        
        # Combine into position embeddings
        position_embeddings = torch.stack([depth_embeddings, sibling_positions], dim=1)
        
        return position_embeddings
    
    def _create_masked_tokens(
        self, 
        tokens: torch.Tensor, 
        attention_mask: torch.Tensor,
        vocab_size: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create masked tokens for diffusion training"""
        masked_tokens = tokens.clone()
        mask_labels = torch.full_like(tokens, -100)
        
        # Only mask valid tokens (not padding)
        valid_positions = attention_mask.nonzero(as_tuple=True)[0]
        
        # Calculate number of tokens to mask
        num_to_mask = int(len(valid_positions) * self.mask_probability)
        
        if num_to_mask > 0:
            # Randomly select positions to mask
            mask_positions = torch.randperm(len(valid_positions))[:num_to_mask]
            mask_indices = valid_positions[mask_positions]
            
            for idx in mask_indices:
                idx = idx.item()
                original_token = tokens[idx].item()
                mask_labels[idx] = original_token
                
                rand = random.random()
                if rand < 0.8:  # 80% of time, replace with [MASK]
                    masked_tokens[idx] = self.mask_token_id
                elif rand < 0.9 and vocab_size:  # 10% of time, replace with random token
                    masked_tokens[idx] = random.randint(2, vocab_size - 1)
        
        return masked_tokens, mask_labels


class LayoutTransforms:
    """
    Layout preprocessing transforms following instruction 2.4:
    - Tokenize structure keys and props entries
    - Create attention masks to enforce valid generation order
    - Label smoothing or class-balanced weighting if element distribution is skewed
    """
    
    def __init__(
        self,
        max_sequence_length: int = 256,
        label_smoothing: float = 0.1,
        pad_token_id: int = 0
    ):
        self.max_sequence_length = max_sequence_length
        self.label_smoothing = label_smoothing
        self.pad_token_id = pad_token_id
    
    def __call__(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply layout transforms"""
        # Step 1: Pad or truncate
        tokens_padded, attention_mask = self._pad_or_truncate(tokens)
        
        # Step 2: Create causal mask for autoregressive generation
        causal_mask = self._create_causal_mask(len(tokens_padded))
        
        # Step 3: Create target labels (shifted tokens)
        labels = self._create_labels(tokens_padded)
        
        # Step 4: Apply label smoothing
        smoothed_labels = self._apply_label_smoothing(labels, tokens_padded)
        
        return {
            'tokens': tokens_padded,
            'attention_mask': attention_mask,
            'causal_mask': causal_mask,
            'labels': labels,
            'smoothed_labels': smoothed_labels
        }
    
    def _pad_or_truncate(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad or truncate to max sequence length"""
        seq_len = len(tokens)
        
        if seq_len >= self.max_sequence_length:
            tokens_padded = tokens[:self.max_sequence_length]
            attention_mask = torch.ones(self.max_sequence_length, dtype=torch.bool)
        else:
            padding_length = self.max_sequence_length - seq_len
            tokens_padded = F.pad(tokens, (0, padding_length), value=self.pad_token_id)
            attention_mask = torch.cat([
                torch.ones(seq_len, dtype=torch.bool),
                torch.zeros(padding_length, dtype=torch.bool)
            ])
        
        return tokens_padded, attention_mask
    
    def _create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Create causal attention mask for autoregressive generation"""
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        return mask
    
    def _create_labels(self, tokens: torch.Tensor) -> torch.Tensor:
        """Create shifted labels for autoregressive training"""
        labels = torch.cat([tokens[1:], torch.tensor([self.pad_token_id])])
        return labels
    
    def _apply_label_smoothing(self, labels: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Apply label smoothing to targets"""
        if self.label_smoothing == 0.0:
            return labels
        
        # Get vocabulary size from max token value
        vocab_size = max(tokens.max().item(), labels.max().item()) + 1
        
        # Create one-hot encoding
        one_hot = F.one_hot(labels, num_classes=vocab_size).float()
        
        # Apply label smoothing
        smooth_labels = one_hot * (1.0 - self.label_smoothing) + \
                       (self.label_smoothing / vocab_size)
        
        return smooth_labels


# Utility classes
class ComposeTransforms:
    """Compose multiple transforms together"""
    
    def __init__(self, transforms: List[Any]):
        self.transforms = transforms
    
    def __call__(self, data: Any) -> Any:
        for transform in self.transforms:
            data = transform(data)
        return data


class ImageAugmentations:
    """Additional image augmentations for training robustness"""
    
    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
        rotation_degrees: float = 5.0,
        apply_probability: float = 0.5
    ):
        self.transform = transforms.Compose([
            transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue
            ),
            transforms.RandomRotation(degrees=rotation_degrees),
        ])
        self.apply_probability = apply_probability
    
    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < self.apply_probability:
            return self.transform(image)
        return image
