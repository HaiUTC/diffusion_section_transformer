"""
Comprehensive Data Loaders for Multimodal Layout Generation
Implements the exact data loading specification from instruction.md
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import re

from .schema import UnifiedExample, UnifiedSchemaValidator
from .filesystem_layout import FilesystemLayoutManager


class VisionLoader:
    """
    Vision data loader for screenshot processing.
    Implements ViT-style patch embedding preprocessing.
    """
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        normalize: bool = True
    ):
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Define preprocessing transforms
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ]
        
        if normalize:
            # ImageNet normalization for ViT compatibility
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )
        
        self.transform = transforms.Compose(transform_list)
    
    def load_screenshot(self, screenshot_path: Path) -> torch.Tensor:
        """
        Load and preprocess a screenshot image.
        
        Args:
            screenshot_path: Path to screenshot image
            
        Returns:
            Preprocessed image tensor [C, H, W]
        """
        try:
            image = Image.open(screenshot_path).convert('RGB')
            return self.transform(image)
        except Exception as e:
            print(f"❌ Error loading screenshot {screenshot_path}: {e}")
            # Return blank image as fallback
            return torch.zeros(3, self.image_size, self.image_size)
    
    def extract_patches(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract ViT-style patches from image.
        
        Args:
            image: Image tensor [C, H, W]
            
        Returns:
            Patch embeddings [num_patches, patch_dim]
        """
        C, H, W = image.shape
        assert H == W == self.image_size, f"Image must be {self.image_size}x{self.image_size}"
        
        # Reshape into patches: [C, H, W] -> [num_patches, patch_dim]
        patches = image.unfold(1, self.patch_size, self.patch_size)  # [C, H/P, W, P]
        patches = patches.unfold(2, self.patch_size, self.patch_size)  # [C, H/P, W/P, P, P]
        patches = patches.permute(1, 2, 0, 3, 4)  # [H/P, W/P, C, P, P]
        patches = patches.reshape(self.num_patches, -1)  # [num_patches, C*P*P]
        
        return patches


class StructuralLoader:
    """
    Structural data loader for HTML hierarchical tokenization.
    Converts HTML structure JSON into token sequences with @ concatenation.
    """
    
    def __init__(
        self,
        vocab_size: int = 4000,
        max_sequence_length: int = 512,
        special_tokens: Optional[Dict[str, str]] = None
    ):
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        
        # Define special tokens
        self.special_tokens = special_tokens or {
            "[PAD]": 0,
            "[UNK]": 1, 
            "[START]": 2,
            "[END]": 3,
            "[SEP]": 4,
            "@": 5  # Concatenation symbol
        }
        
        # Initialize vocabulary mapping
        self.token_to_id = self.special_tokens.copy()
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}
        self.next_id = max(self.special_tokens.values()) + 1
    
    def build_vocabulary(self, structure_data_list: List[Dict[str, Any]]):
        """
        Build vocabulary from structure data.
        
        Args:
            structure_data_list: List of HTML structure dictionaries
        """
        all_tokens = set()
        
        for structure_data in structure_data_list:
            tokens = self._extract_tokens_from_structure(structure_data)
            all_tokens.update(tokens)
        
        # Add most frequent tokens to vocabulary
        sorted_tokens = sorted(all_tokens)[:self.vocab_size - len(self.special_tokens)]
        
        for token in sorted_tokens:
            if token not in self.token_to_id:
                self.token_to_id[token] = self.next_id
                self.id_to_token[self.next_id] = token
                self.next_id += 1
        
        print(f"✅ Built vocabulary with {len(self.token_to_id)} tokens")
    
    def _extract_tokens_from_structure(self, structure: Dict[str, Any], prefix: str = "") -> List[str]:
        """
        Recursively extract tokens from HTML structure.
        
        Args:
            structure: HTML structure dictionary
            prefix: Current hierarchical prefix
            
        Returns:
            List of extracted tokens
        """
        tokens = []
        
        for key, value in structure.items():
            # Add element token
            full_key = f"{prefix}@{key}" if prefix else key
            tokens.append(full_key)
            
            # Process element attributes/text
            if isinstance(value, dict):
                # Check for text content
                if "text" in value:
                    text_tokens = self._tokenize_text(value["text"])
                    tokens.extend(text_tokens)
                
                # Process nested structure
                nested_tokens = self._extract_tokens_from_structure(value, full_key)
                tokens.extend(nested_tokens)
            elif isinstance(value, str):
                # Direct text content
                text_tokens = self._tokenize_text(value)
                tokens.extend(text_tokens)
        
        return tokens
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Simple text tokenization."""
        # Basic tokenization (can be enhanced with proper NLP tokenizer)
        tokens = re.findall(r'\w+|[^\w\s]', text.lower())
        return tokens
    
    def structure_to_tokens(self, structure_data: Dict[str, Any]) -> torch.Tensor:
        """
        Convert HTML structure to token sequence.
        
        Args:
            structure_data: HTML structure dictionary
            
        Returns:
            Token sequence tensor [sequence_length]
        """
        # Extract tokens
        tokens = self._extract_tokens_from_structure(structure_data.get("data", {}))
        
        # Convert to IDs
        token_ids = [self.special_tokens["[START]"]]
        
        for token in tokens[:self.max_sequence_length - 2]:  # Reserve space for START/END
            token_id = self.token_to_id.get(token, self.special_tokens["[UNK]"])
            token_ids.append(token_id)
        
        token_ids.append(self.special_tokens["[END]"])
        
        # Pad to max length
        while len(token_ids) < self.max_sequence_length:
            token_ids.append(self.special_tokens["[PAD]"])
        
        return torch.tensor(token_ids[:self.max_sequence_length], dtype=torch.long)
    
    def create_attention_mask(self, token_sequence: torch.Tensor) -> torch.Tensor:
        """
        Create attention mask for token sequence.
        
        Args:
            token_sequence: Token sequence tensor
            
        Returns:
            Attention mask tensor [sequence_length]
        """
        pad_token_id = self.special_tokens["[PAD]"]
        return (token_sequence != pad_token_id).long()


class LabelLoader:
    """
    Label data loader for layout tokenization.
    Converts SectionLayout JSON into target token sequences.
    """
    
    def __init__(
        self,
        element_vocab_size: int = 200,
        property_vocab_size: int = 100,
        max_elements: int = 32
    ):
        self.element_vocab_size = element_vocab_size
        self.property_vocab_size = property_vocab_size
        self.max_elements = max_elements
        
        # Special tokens for layout
        self.special_element_tokens = {
            "[PAD]": 0,
            "[UNK]": 1,
            "section": 2,
            "heading": 3,
            "paragraph": 4,
            "button": 5,
            "image": 6,
            "container": 7
        }
        
        self.special_property_tokens = {
            "[PAD]": 0,
            "[UNK]": 1,
            "bi": 2,  # background image
            "bo": 3,  # background overlay
            "bv": 4   # background video
        }
        
        # Initialize vocabularies
        self.element_to_id = self.special_element_tokens.copy()
        self.property_to_id = self.special_property_tokens.copy()
        self.next_element_id = max(self.special_element_tokens.values()) + 1
        self.next_property_id = max(self.special_property_tokens.values()) + 1
    
    def build_label_vocabulary(self, layout_data_list: List[Dict[str, Any]]):
        """
        Build vocabulary from layout data.
        
        Args:
            layout_data_list: List of layout dictionaries
        """
        all_elements = set()
        all_properties = set()
        
        for layout_data in layout_data_list:
            elements, properties = self._extract_layout_tokens(layout_data)
            all_elements.update(elements)
            all_properties.update(properties)
        
        # Add elements to vocabulary
        sorted_elements = sorted(all_elements)[:self.element_vocab_size - len(self.special_element_tokens)]
        for element in sorted_elements:
            if element not in self.element_to_id:
                self.element_to_id[element] = self.next_element_id
                self.next_element_id += 1
        
        # Add properties to vocabulary
        sorted_properties = sorted(all_properties)[:self.property_vocab_size - len(self.special_property_tokens)]
        for prop in sorted_properties:
            if prop not in self.property_to_id:
                self.property_to_id[prop] = self.next_property_id
                self.next_property_id += 1
        
        print(f"✅ Built label vocabulary: {len(self.element_to_id)} elements, {len(self.property_to_id)} properties")
    
    def _extract_layout_tokens(self, layout_data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """
        Extract element and property tokens from layout data.
        
        Args:
            layout_data: Layout dictionary
            
        Returns:
            (element_tokens, property_tokens)
        """
        elements = []
        properties = []
        
        # Extract from structure
        structure = layout_data.get("data", {}).get("structure", {})
        elements.extend(self._extract_elements_from_structure(structure))
        
        # Extract from properties
        props = layout_data.get("props", {})
        properties.extend(props.keys())
        
        return elements, properties
    
    def _extract_elements_from_structure(self, structure: Dict[str, Any]) -> List[str]:
        """Extract element types from @ concatenation syntax."""
        elements = []
        
        for key, value in structure.items():
            if "@" in key:
                # Extract element type from @ concatenation
                parts = key.split("@")
                element_type = parts[0]  # e.g., "section" from "section@div.container"
                elements.append(element_type)
            
            # Recursively process nested structures
            if isinstance(value, dict):
                nested_elements = self._extract_elements_from_structure(value)
                elements.extend(nested_elements)
        
        return elements
    
    def layout_to_tokens(self, layout_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Convert layout data to tokenized format.
        
        Args:
            layout_data: Layout dictionary
            
        Returns:
            Dictionary with tokenized layout components
        """
        # Extract elements and properties
        elements, properties = self._extract_layout_tokens(layout_data)
        
        # Convert elements to IDs
        element_ids = []
        for element in elements[:self.max_elements]:
            element_id = self.element_to_id.get(element, self.special_element_tokens["[UNK]"])
            element_ids.append(element_id)
        
        # Pad elements
        while len(element_ids) < self.max_elements:
            element_ids.append(self.special_element_tokens["[PAD]"])
        
        # Convert properties to IDs
        property_ids = []
        for prop in properties:
            prop_id = self.property_to_id.get(prop, self.special_property_tokens["[UNK]"])
            property_ids.append(prop_id)
        
        # Create one-hot for properties (multi-label)
        property_vector = torch.zeros(self.property_vocab_size)
        for prop_id in property_ids:
            if prop_id < self.property_vocab_size:
                property_vector[prop_id] = 1.0
        
        return {
            "element_tokens": torch.tensor(element_ids[:self.max_elements], dtype=torch.long),
            "property_tokens": property_vector,
            "num_elements": torch.tensor(min(len(elements), self.max_elements), dtype=torch.long)
        }


class MultimodalLayoutDataset(Dataset):
    """
    Complete multimodal dataset for layout generation.
    Combines vision, structure, and layout data loaders.
    """
    
    def __init__(
        self,
        filesystem_manager: FilesystemLayoutManager,
        split: str = "train",
        image_size: int = 224,
        patch_size: int = 16,
        structure_vocab_size: int = 4000,
        structure_max_length: int = 512,
        element_vocab_size: int = 200,
        property_vocab_size: int = 100,
        max_elements: int = 32,
        augment: bool = False
    ):
        self.filesystem_manager = filesystem_manager
        self.split = split
        self.augment = augment
        
        # Initialize loaders
        self.vision_loader = VisionLoader(image_size, patch_size)
        self.structural_loader = StructuralLoader(structure_vocab_size, structure_max_length)
        self.label_loader = LabelLoader(element_vocab_size, property_vocab_size, max_elements)
        
        # Get example IDs for this split
        self.example_ids = filesystem_manager.list_examples(split)
        
        if not self.example_ids:
            print(f"⚠️ Warning: No examples found for split '{split}'")
        
        print(f"✅ Loaded {len(self.example_ids)} examples for {split} split")
    
    def build_vocabularies(self, splits: List[str] = ["train"]):
        """
        Build vocabularies from training data.
        
        Args:
            splits: List of splits to use for vocabulary building
        """
        all_structures = []
        all_layouts = []
        
        for split in splits:
            example_ids = self.filesystem_manager.list_examples(split)
            
            for example_id in example_ids:
                try:
                    example_path = self.filesystem_manager.get_example_path(example_id)
                    example, errors = UnifiedSchemaValidator.load_and_validate_example(example_path)
                    
                    if example:
                        all_structures.append(example.structure.__dict__)
                        all_layouts.append(example.layout.__dict__)
                        
                except Exception as e:
                    print(f"❌ Error loading example {example_id}: {e}")
                    continue
        
        # Build vocabularies
        self.structural_loader.build_vocabulary(all_structures)
        self.label_loader.build_label_vocabulary(all_layouts)
        
        print(f"✅ Built vocabularies from {len(all_structures)} examples")
    
    def __len__(self) -> int:
        return len(self.example_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example.
        
        Args:
            idx: Example index
            
        Returns:
            Dictionary with multimodal data
        """
        example_id = self.example_ids[idx]
        
        try:
            # Load example
            example_path = self.filesystem_manager.get_example_path(example_id)
            example, errors = UnifiedSchemaValidator.load_and_validate_example(example_path)
            
            if not example:
                print(f"❌ Failed to load example {example_id}: {errors}")
                return self._get_empty_example()
            
            # Load screenshot
            screenshot_path = self.filesystem_manager.get_screenshot_path(example.screenshot.path)
            screenshot_tensor = self.vision_loader.load_screenshot(screenshot_path)
            
            # Process structure
            structure_tokens = self.structural_loader.structure_to_tokens(example.structure.__dict__)
            structure_mask = self.structural_loader.create_attention_mask(structure_tokens)
            
            # Process layout
            layout_tokens = self.label_loader.layout_to_tokens(example.layout.__dict__)
            
            return {
                "screenshot": screenshot_tensor,
                "structure_tokens": structure_tokens,
                "structure_mask": structure_mask,
                "element_tokens": layout_tokens["element_tokens"],
                "property_tokens": layout_tokens["property_tokens"],
                "num_elements": layout_tokens["num_elements"],
                "example_id": example_id
            }
            
        except Exception as e:
            print(f"❌ Error processing example {example_id}: {e}")
            return self._get_empty_example()
    
    def _get_empty_example(self) -> Dict[str, torch.Tensor]:
        """Return empty example as fallback."""
        return {
            "screenshot": torch.zeros(3, self.vision_loader.image_size, self.vision_loader.image_size),
            "structure_tokens": torch.zeros(self.structural_loader.max_sequence_length, dtype=torch.long),
            "structure_mask": torch.zeros(self.structural_loader.max_sequence_length, dtype=torch.long),
            "element_tokens": torch.zeros(self.label_loader.max_elements, dtype=torch.long),
            "property_tokens": torch.zeros(self.label_loader.property_vocab_size),
            "num_elements": torch.tensor(0, dtype=torch.long),
            "example_id": "empty"
        }


def create_data_loaders(
    filesystem_manager: FilesystemLayoutManager,
    batch_size: int = 8,
    num_workers: int = 4,
    **kwargs
) -> Dict[str, DataLoader]:
    """
    Create data loaders for all splits.
    
    Args:
        filesystem_manager: Filesystem manager instance
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        **kwargs: Additional arguments for dataset
        
    Returns:
        Dictionary mapping split names to DataLoader instances
    """
    data_loaders = {}
    
    # Get available splits
    manifest = filesystem_manager.load_manifest()
    if manifest is None:
        print("❌ No manifest found")
        return {}
    
    available_splits = [name for name, split in manifest.splits.items() if split.size > 0]
    
    # Create dataset for vocabulary building (use train split if available)
    vocab_splits = ["train"] if "train" in available_splits else available_splits[:1]
    vocab_dataset = MultimodalLayoutDataset(filesystem_manager, vocab_splits[0], **kwargs)
    vocab_dataset.build_vocabularies(vocab_splits)
    
    # Create data loaders for each split
    for split_name in available_splits:
        dataset = MultimodalLayoutDataset(filesystem_manager, split_name, **kwargs)
        
        # Copy vocabularies from the vocab_dataset
        dataset.structural_loader = vocab_dataset.structural_loader
        dataset.label_loader = vocab_dataset.label_loader
        
        # Set shuffle based on split
        shuffle = (split_name == "train")
        
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split_name == "train")  # Drop last batch only for training
        )
        
        data_loaders[split_name] = data_loader
        print(f"✅ Created {split_name} data loader: {len(dataset)} examples, batch_size={batch_size}")
    
    return data_loaders 