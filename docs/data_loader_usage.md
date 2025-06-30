# Data Loader Module Usage Guide

This guide explains how to use the data loader modules for your Generative AI engine that transforms screenshots and HTML structures into structured section layouts.

## Overview

The data loader implementation consists of three main components:

1. **VisionLoader** - Processes screenshot images into ViT-style patch embeddings
2. **StructureLoader** - Converts HTML structure objects into hierarchical token sequences
3. **LabelLoader** - Processes layout JSON into target token sequences for training
4. **DatasetLoader** - Coordinates all loaders for complete dataset processing

## Quick Start

```python
from data_loaders import DatasetLoader, validate_dataset

# Initialize dataset loader
dataset_loader = DatasetLoader(
    dataset_root="path/to/your/dataset",
    split="train",
    patch_size=16,
    target_size=512
)

# Load a single example
example = dataset_loader[0]
print(f"Vision patches: {example['vision_patches'].shape}")
print(f"Structure tokens: {example['structure_tokens'].shape}")
print(f"Label tokens: {example['label_tokens'].shape}")
```

## Individual Loader Usage

### VisionLoader

Processes screenshot images into patch embeddings for Vision Transformer models:

```python
from data_loaders import VisionLoader

vision_loader = VisionLoader(patch_size=16, target_size=512)
patches = vision_loader.load_and_process("screenshot.png", width=1920, height=824)

# Output: torch.Tensor of shape (num_patches, patch_dim)
# For 512x512 image with 16x16 patches: (1024, 768)
```

**Key Features:**

- Resizes images to target size (default: 512x512)
- Converts to non-overlapping patches (default: 16x16)
- Applies ImageNet normalization
- Returns patch embeddings ready for transformer input

### StructureLoader

Converts nested HTML structure objects into token sequences with hierarchy information:

```python
from data_loaders import StructureLoader

structure_loader = StructureLoader()
structure_data = {
    "div.container": {
        "h1.heading": {"text": "Hello World"},
        "p.paragraph": {"text": "This is a paragraph"}
    }
}

tokens, hierarchy = structure_loader.load_and_process(structure_data)

# Output:
# - tokens: torch.Tensor of token IDs
# - hierarchy: torch.Tensor of (depth, sibling_index) pairs
```

**Key Features:**

- Traverses nested structures in preorder
- Handles compound keys with @ concatenation
- Includes hierarchy position embeddings
- Extensible vocabulary with special tokens

### LabelLoader

Processes layout JSON into target sequences for sequence-to-sequence training:

```python
from data_loaders import LabelLoader

label_loader = LabelLoader()
layout_data = {
    "structure": {
        "section@div.container": {
            "heading@h1.heading": "",
            "paragraph@p.paragraph": ""
        }
    },
    "props": {
        "bi": "div.background_image"
    }
}

label_tokens = label_loader.load_and_process(layout_data)

# Output: torch.Tensor of target token sequence
```

**Key Features:**

- Handles @ concatenation syntax for element merging
- Processes both structure and props
- Special tokens for props delimiting
- Column indexing support (column$1, column$2, etc.)

## Dataset Structure

Your dataset should follow this structure:

```
dataset_root/
├── train/
│   ├── example_0001/
│   │   ├── screenshot.png
│   │   └── example.json
│   └── example_0002/
│       └── ...
├── val/
│   └── ...
├── test/
│   └── ...
└── dataset_config.yaml
```

## Testing & Validation

### Test Commands

Run comprehensive dataset pipeline tests:

```bash
# Test all data loader components
python examples/test_dataset_pipeline_demo.py

# Validate your dataset
python scripts/validate_dataset.py /path/to/dataset

# Validate dataset with detailed output
python scripts/validate_dataset.py /path/to/dataset --verbose

# Auto-exclude failed examples
python scripts/validate_dataset.py /path/to/dataset --auto-exclude
```

### Expected Results

The dataset pipeline tests validate:

- **✅ Unified Schema**: JSON schema compliance and validation
- **✅ Filesystem Layout**: Directory structure and manifest creation
- **✅ Data Loaders**: Vision, structure, and label processing
- **✅ Complete Integration**: End-to-end multimodal data loading

### Example JSON Format

```json
{
  "id": "unique_example_id",
  "screenshot": {
    "path": "screenshot.png",
    "width": 1920,
    "height": 824
  },
  "structure": {
    "type": "HTMLObject",
    "data": {
      "div.container": {
        "h1.heading": { "text": "Hello World" }
      }
    }
  },
  "layout": {
    "type": "SectionLayout",
    "data": {
      "structure": {
        "section@div.container": {
          "heading@h1.heading": ""
        }
      },
      "props": {}
    }
  }
}
```

### Dataset Config YAML

```yaml
splits:
  train:
    - example_0001
    - example_0002
  val:
    - example_1001
  test:
    - example_2001
```

## Dataset Validation

Before training, validate your dataset:

```python
from data_loaders import validate_dataset

results = validate_dataset("path/to/dataset")

print(f"Valid examples: {len(results['valid_examples'])}")
print(f"Missing files: {len(results['missing_files'])}")
print(f"Invalid JSON: {len(results['invalid_json'])}")
print(f"Syntax errors: {len(results['syntax_errors'])}")
```

The validator checks:

- File existence (screenshot.png and example.json)
- JSON schema compliance
- Vocabulary consistency (all tokens in predefined vocabularies)
- Syntax correctness (@ concatenation and colon-delimited props)

## Advanced Features

### Batch Processing

```python
from torch.utils.data import DataLoader

# Create PyTorch DataLoader for training
dataloader = DataLoader(
    dataset_loader,
    batch_size=8,
    shuffle=True,
    collate_fn=dataset_loader.collate_fn
)

for batch in dataloader:
    vision_patches = batch['vision_patches']  # [batch, patches, dim]
    structure_tokens = batch['structure_tokens']  # [batch, seq_len]
    label_tokens = batch['label_tokens']  # [batch, label_seq_len]

    # Training loop here
    break
```

### Custom Transforms

```python
from torchvision import transforms

# Custom image transforms
custom_transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

vision_loader = VisionLoader(
    patch_size=16,
    target_size=512,
    transforms=custom_transforms
)
```

### Vocabulary Management

```python
# Access vocabularies
structure_vocab = structure_loader.get_vocabulary()
label_vocab = label_loader.get_vocabulary()

print(f"Structure vocab size: {len(structure_vocab)}")
print(f"Label vocab size: {len(label_vocab)}")

# Add custom tokens
structure_loader.add_tokens(['custom_element', 'special_token'])
```

## Performance Optimization

### Caching

The data loaders implement intelligent caching:

- **Preprocessed patches** cached to disk
- **Tokenized structures** cached in memory
- **Vocabulary mappings** cached for reuse

### Memory Management

```python
# Configure memory usage
dataset_loader = DatasetLoader(
    dataset_root="path/to/dataset",
    split="train",
    cache_size=1000,  # Number of examples to cache
    preload=False     # Disable preloading for large datasets
)
```

### Parallel Loading

```python
# Use multiple workers for faster loading
dataloader = DataLoader(
    dataset_loader,
    batch_size=8,
    num_workers=4,  # Parallel loading
    pin_memory=True  # Faster GPU transfer
)
```

## Integration with Model Training

The data loaders are designed to integrate seamlessly with transformer models:

```python
# Compatible with any PyTorch model
model = YourMultimodalTransformer()

for batch in dataloader:
    # Forward pass
    outputs = model(
        vision_input=batch['vision_patches'],
        structure_input=batch['structure_tokens'],
        targets=batch['label_tokens']
    )

    # Loss computation
    loss = criterion(outputs, batch['label_tokens'])
    loss.backward()
```

## Troubleshooting

### Common Issues

1. **Image Loading Errors**: Check file permissions and formats
2. **Vocabulary Mismatches**: Ensure consistent tokenization
3. **Memory Issues**: Reduce batch size or enable lazy loading
4. **Performance**: Enable caching and use multiple workers

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Verbose dataset loading
dataset_loader = DatasetLoader(
    dataset_root="path/to/dataset",
    split="train",
    verbose=True
)
```

---

For complete implementation details, see the Step 2 completion summary in `docs/step2_completion_summary.md`.
