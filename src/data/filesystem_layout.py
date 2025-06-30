"""
Filesystem Layout Management for Multimodal Dataset
Implements the exact filesystem structure specified in instruction.md
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from .schema import UnifiedExample, UnifiedSchemaValidator


@dataclass
class DatasetSplit:
    """Dataset split configuration."""
    name: str
    size: int
    examples: List[str]  # List of example IDs


@dataclass
class DatasetManifest:
    """
    Dataset manifest schema as specified in instruction.md.
    
    Example dataset_config.yaml:
    metadata:
      name: "section_layout_dataset"
      version: "1.0.0"
      description: "HTML/CSS to layout conversion dataset"
      total_examples: 1500
      created_date: "2024-01-15"
      
    splits:
      train:
        size: 1200
        examples: ["example_001", "example_002", ...]
      validation:
        size: 200
        examples: ["example_1201", "example_1202", ...]
      test:
        size: 100
        examples: ["example_1401", "example_1402", ...]
        
    schema_version: "1.0"
    paths:
      base_dir: "data/processed"
      examples_dir: "examples"
      screenshots_dir: "screenshots"
    """
    metadata: Dict[str, Any]
    splits: Dict[str, DatasetSplit]
    schema_version: str
    paths: Dict[str, str]


class FilesystemLayoutManager:
    """
    Manages the filesystem layout for the multimodal dataset.
    
    Directory structure:
    data/
    ├── raw/                    # Raw data before processing
    ├── processed/              # Processed dataset
    │   ├── examples/           # JSON example files
    │   ├── screenshots/        # Screenshot images
    │   └── dataset_config.yaml # Dataset manifest
    ├── cache/                  # Preprocessing cache
    └── validation/             # Validation reports
    """
    
    def __init__(self, base_data_dir: str = "data"):
        self.base_dir = Path(base_data_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        self.cache_dir = self.base_dir / "cache"
        self.validation_dir = self.base_dir / "validation"
        
        # Processed subdirectories
        self.examples_dir = self.processed_dir / "examples"
        self.screenshots_dir = self.processed_dir / "screenshots"
        
        # Manifest file
        self.manifest_path = self.processed_dir / "dataset_config.yaml"
    
    def create_directory_structure(self):
        """Create the complete directory structure."""
        directories = [
            self.raw_dir,
            self.processed_dir,
            self.examples_dir,
            self.screenshots_dir,
            self.cache_dir,
            self.validation_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        print(f"✅ Created dataset directory structure at: {self.base_dir}")
    
    def create_manifest_template(
        self,
        dataset_name: str = "section_layout_dataset",
        description: str = "HTML/CSS to layout conversion dataset"
    ) -> DatasetManifest:
        """Create a template dataset manifest."""
        
        manifest = DatasetManifest(
            metadata={
                "name": dataset_name,
                "version": "1.0.0",
                "description": description,
                "total_examples": 0,
                "created_date": "2024-01-15"
            },
            splits={
                "train": DatasetSplit("train", 0, []),
                "validation": DatasetSplit("validation", 0, []),
                "test": DatasetSplit("test", 0, [])
            },
            schema_version="1.0",
            paths={
                "base_dir": str(self.processed_dir.relative_to(self.base_dir)),
                "examples_dir": "examples",
                "screenshots_dir": "screenshots"
            }
        )
        
        return manifest
    
    def save_manifest(self, manifest: DatasetManifest):
        """Save dataset manifest to YAML file."""
        
        # Convert DatasetSplit objects to dictionaries
        manifest_dict = {
            "metadata": manifest.metadata,
            "splits": {
                name: {
                    "size": split.size,
                    "examples": split.examples
                }
                for name, split in manifest.splits.items()
            },
            "schema_version": manifest.schema_version,
            "paths": manifest.paths
        }
        
        with open(self.manifest_path, 'w') as f:
            yaml.dump(manifest_dict, f, default_flow_style=False, indent=2)
        
        print(f"✅ Saved dataset manifest to: {self.manifest_path}")
    
    def load_manifest(self) -> Optional[DatasetManifest]:
        """Load dataset manifest from YAML file."""
        if not self.manifest_path.exists():
            return None
        
        try:
            with open(self.manifest_path, 'r') as f:
                data = yaml.safe_load(f)
            
            # Convert dictionary back to DatasetManifest
            splits = {}
            for name, split_data in data['splits'].items():
                splits[name] = DatasetSplit(
                    name=name,
                    size=split_data['size'],
                    examples=split_data['examples']
                )
            
            manifest = DatasetManifest(
                metadata=data['metadata'],
                splits=splits,
                schema_version=data['schema_version'],
                paths=data['paths']
            )
            
            return manifest
            
        except Exception as e:
            print(f"❌ Error loading manifest: {e}")
            return None
    
    def add_example(self, example: UnifiedExample, split: str = "train") -> bool:
        """
        Add a new example to the dataset.
        
        Args:
            example: The example to add
            split: Which split to add to (train/validation/test)
            
        Returns:
            Success status
        """
        try:
            # Validate example
            example_dict = asdict(example)
            is_valid, errors = UnifiedSchemaValidator.validate_example(example_dict)
            
            if not is_valid:
                print(f"❌ Example validation failed: {errors}")
                return False
            
            # Save example JSON
            example_path = self.examples_dir / f"{example.id}.json"
            with open(example_path, 'w') as f:
                json.dump(example_dict, f, indent=2)
            
            # Update manifest
            manifest = self.load_manifest()
            if manifest is None:
                manifest = self.create_manifest_template()
            
            # Add to appropriate split
            if split not in manifest.splits:
                manifest.splits[split] = DatasetSplit(split, 0, [])
            
            if example.id not in manifest.splits[split].examples:
                manifest.splits[split].examples.append(example.id)
                manifest.splits[split].size += 1
                manifest.metadata['total_examples'] = sum(
                    split.size for split in manifest.splits.values()
                )
            
            self.save_manifest(manifest)
            
            print(f"✅ Added example {example.id} to {split} split")
            return True
            
        except Exception as e:
            print(f"❌ Error adding example: {e}")
            return False
    
    def get_example_path(self, example_id: str) -> Path:
        """Get the file path for an example JSON."""
        return self.examples_dir / f"{example_id}.json"
    
    def get_screenshot_path(self, screenshot_filename: str) -> Path:
        """Get the file path for a screenshot."""
        return self.screenshots_dir / screenshot_filename
    
    def list_examples(self, split: Optional[str] = None) -> List[str]:
        """List all example IDs, optionally filtered by split."""
        manifest = self.load_manifest()
        if manifest is None:
            return []
        
        if split:
            return manifest.splits.get(split, DatasetSplit(split, 0, [])).examples
        else:
            all_examples = []
            for split_data in manifest.splits.values():
                all_examples.extend(split_data.examples)
            return all_examples
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics."""
        manifest = self.load_manifest()
        if manifest is None:
            return {"error": "No manifest found"}
        
        stats = {
            "metadata": manifest.metadata,
            "total_examples": manifest.metadata.get('total_examples', 0),
            "splits": {
                name: split.size for name, split in manifest.splits.items()
            },
            "directory_structure": {
                "base_dir": str(self.base_dir),
                "examples_dir": str(self.examples_dir),
                "screenshots_dir": str(self.screenshots_dir),
                "manifest_path": str(self.manifest_path)
            }
        }
        
        # Count actual files
        if self.examples_dir.exists():
            actual_examples = len(list(self.examples_dir.glob("*.json")))
            stats["actual_files"] = {
                "examples": actual_examples,
                "screenshots": len(list(self.screenshots_dir.glob("*"))) if self.screenshots_dir.exists() else 0
            }
        
        return stats
    
    def create_split_indices(
        self,
        total_examples: int,
        train_ratio: float = 0.8,
        val_ratio: float = 0.15,
        test_ratio: float = 0.05
    ) -> Dict[str, List[str]]:
        """
        Create balanced split indices for dataset organization.
        
        Args:
            total_examples: Total number of examples
            train_ratio: Ratio for training split
            val_ratio: Ratio for validation split
            test_ratio: Ratio for test split
            
        Returns:
            Dictionary mapping split names to example ID lists
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        # Calculate split sizes
        train_size = int(total_examples * train_ratio)
        val_size = int(total_examples * val_ratio)
        test_size = total_examples - train_size - val_size  # Remainder goes to test
        
        # Generate example IDs
        example_ids = [f"example_{i:06d}" for i in range(1, total_examples + 1)]
        
        # Create splits
        splits = {
            "train": example_ids[:train_size],
            "validation": example_ids[train_size:train_size + val_size],
            "test": example_ids[train_size + val_size:]
        }
        
        return splits


def setup_dataset_structure(
    base_dir: str = "data",
    dataset_name: str = "section_layout_dataset"
) -> FilesystemLayoutManager:
    """
    Setup complete dataset structure and return manager.
    
    Args:
        base_dir: Base directory for dataset
        dataset_name: Name of the dataset
        
    Returns:
        Configured FilesystemLayoutManager
    """
    manager = FilesystemLayoutManager(base_dir)
    manager.create_directory_structure()
    
    # Create initial manifest if it doesn't exist
    if not manager.manifest_path.exists():
        manifest = manager.create_manifest_template(dataset_name)
        manager.save_manifest(manifest)
    
    return manager 