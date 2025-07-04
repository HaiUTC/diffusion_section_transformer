#!/usr/bin/env python3
"""
Dataset Creation Script for Diffusion Section Transformer

This script converts Hugging Face datasets to the expected training format:
- Downloads dataset from Hugging Face
- Converts to directory structure with train/val/test splits
- Each example contains: screenshot.png, structure.json, layout.json

Expected HuggingFace dataset format:
- 'screenshot': PIL Image or image path
- 'structure': JSON string or dict containing HTML/CSS structure
- 'layout': JSON string or dict containing layout information

Usage:
    # From Hugging Face dataset
    python3 scripts/create_dataset.py \
      --dataset_name "username/dataset-name" \
      --output_dir data/raw \
      --num_samples 5000 \
      --split_ratio 0.8,0.1,0.1
    
    # From local dataset
    python3 scripts/create_dataset.py \
      --dataset_path /path/to/local/dataset \
      --output_dir data/raw \
      --num_samples 1000 \
      --split_ratio 0.7,0.15,0.15
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import random
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from datasets import load_dataset, Dataset
    from PIL import Image
except ImportError as e:
    print(f"❌ Missing required packages. Please install:")
    print(f"   pip install datasets pillow")
    print(f"Error: {e}")
    sys.exit(1)


class DatasetConverter:
    """Converts Hugging Face datasets to training format."""
    
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Parse split ratios
        if args.split_ratio:
            ratios = [float(x) for x in args.split_ratio.split(',')]
            if len(ratios) != 3 or not abs(sum(ratios) - 1.0) < 1e-6:
                raise ValueError("Split ratios must sum to 1.0 and contain 3 values")
            self.train_ratio, self.val_ratio, self.test_ratio = ratios
        else:
            self.train_ratio, self.val_ratio, self.test_ratio = 0.8, 0.1, 0.1
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'successful_conversions': 0,
            'failed_conversions': 0,
            'splits': {'train': 0, 'val': 0, 'test': 0}
        }
    
    def load_dataset(self) -> Dataset:
        """Load dataset from Hugging Face or local path."""
        print("📥 Loading dataset...")
        
        if self.args.dataset_name:
            print(f"🌐 Loading from Hugging Face: {self.args.dataset_name}")
            try:
                # Load from Hugging Face
                dataset = load_dataset(
                    self.args.dataset_name,
                    split=self.args.dataset_split,
                    streaming=self.args.streaming
                )
                print(f"✅ Successfully loaded dataset")
                
                # Convert to regular dataset if streaming
                if self.args.streaming:
                    print("🔄 Converting streaming dataset to regular dataset...")
                    dataset = Dataset.from_generator(
                        lambda: dataset,
                        features=dataset.features
                    )
                
            except Exception as e:
                print(f"❌ Failed to load dataset from Hugging Face: {e}")
                print("💡 Make sure the dataset exists and you have proper access")
                sys.exit(1)
                
        elif self.args.dataset_path:
            print(f"📁 Loading from local path: {self.args.dataset_path}")
            try:
                if os.path.isfile(self.args.dataset_path):
                    # Load from file (JSON, CSV, etc.)
                    if self.args.dataset_path.endswith('.json'):
                        dataset = Dataset.from_json(self.args.dataset_path)
                    elif self.args.dataset_path.endswith('.csv'):
                        dataset = Dataset.from_csv(self.args.dataset_path)
                    else:
                        raise ValueError("Unsupported file format. Use .json or .csv")
                else:
                    # Load from directory
                    dataset = Dataset.load_from_disk(self.args.dataset_path)
                
                print(f"✅ Successfully loaded local dataset")
                
            except Exception as e:
                print(f"❌ Failed to load local dataset: {e}")
                sys.exit(1)
        else:
            raise ValueError("Must specify either --dataset_name or --dataset_path")
        
        # Limit dataset size if specified
        if self.args.num_samples and len(dataset) > self.args.num_samples:
            print(f"📊 Limiting dataset to {self.args.num_samples} samples (from {len(dataset)})")
            # Shuffle and select subset
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            indices = indices[:self.args.num_samples]
            dataset = dataset.select(indices)
        
        print(f"📊 Dataset size: {len(dataset):,} samples")
        print(f"📋 Dataset features: {list(dataset.features.keys())}")
        
        return dataset
    
    def validate_dataset_format(self, dataset: Dataset) -> bool:
        """Validate that dataset has required columns."""
        required_columns = ['image', 'structure', 'layout']
        available_columns = list(dataset.features.keys())
        
        missing_columns = [col for col in required_columns if col not in available_columns]
        
        if missing_columns:
            print(f"❌ Dataset missing required columns: {missing_columns}")
            print(f"📋 Available columns: {available_columns}")
            
            # Suggest alternative column names
            suggestions = {}
            for missing in missing_columns:
                for available in available_columns:
                    if missing.lower() in available.lower() or available.lower() in missing.lower():
                        suggestions[missing] = available
            
            if suggestions:
                print(f"💡 Suggested column mappings:")
                for required, suggested in suggestions.items():
                    print(f"   {required} -> {suggested}")
                print("Use --column_mapping to specify custom mappings")
            
            return False
        
        print(f"✅ Dataset format validated")
        return True
    
    def create_splits(self, dataset: Dataset) -> Tuple[Dataset, Dataset, Dataset]:
        """Create train/validation/test splits."""
        print("🔀 Creating dataset splits...")
        
        total_size = len(dataset)
        train_size = int(total_size * self.train_ratio)
        val_size = int(total_size * self.val_ratio)
        test_size = total_size - train_size - val_size
        
        print(f"📊 Split sizes:")
        print(f"   Train: {train_size:,} ({self.train_ratio:.1%})")
        print(f"   Val:   {val_size:,} ({self.val_ratio:.1%})")
        print(f"   Test:  {test_size:,} ({self.test_ratio:.1%})")
        
        # Shuffle indices
        indices = list(range(total_size))
        random.shuffle(indices)
        
        # Create splits
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        train_dataset = dataset.select(train_indices)
        val_dataset = dataset.select(val_indices)
        test_dataset = dataset.select(test_indices)
        
        return train_dataset, val_dataset, test_dataset
    
    def process_example(self, example: Dict, example_id: str, split_dir: Path) -> bool:
        """Process a single example and save to disk."""
        try:
            example_dir = split_dir / example_id
            example_dir.mkdir(parents=True, exist_ok=True)
            
            # Process screenshot
            screenshot = example['image']
            if isinstance(screenshot, str):
                # If it's a path, load the image
                screenshot = Image.open(screenshot)
            elif hasattr(screenshot, 'save'):
                # It's already a PIL Image
                pass
            else:
                # Try to convert to PIL Image
                screenshot = Image.fromarray(screenshot)
            
            # Save screenshot
            screenshot_path = example_dir / "screenshot.png"
            if screenshot.mode != 'RGB':
                screenshot = screenshot.convert('RGB')
            screenshot.save(screenshot_path)
            
            # Process structure
            structure = example['structure']
            if isinstance(structure, str):
                try:
                    structure = json.loads(structure)
                except json.JSONDecodeError:
                    # If it's not valid JSON, wrap it as a simple structure
                    structure = {"content": structure}
            
            # Save structure.json
            structure_path = example_dir / "structure.json"
            with open(structure_path, 'w', encoding='utf-8') as f:
                json.dump(structure, f, indent=2, ensure_ascii=False)
            
            # Process layout
            layout = example['layout']
            if isinstance(layout, str):
                try:
                    layout = json.loads(layout)
                except json.JSONDecodeError:
                    # If it's not valid JSON, wrap it as a simple layout
                    layout = {"elements": [{"type": "unknown", "content": layout}]}
            
            # Save layout.json
            layout_path = example_dir / "layout.json"
            with open(layout_path, 'w', encoding='utf-8') as f:
                json.dump(layout, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"⚠️ Failed to process example {example_id}: {e}")
            return False
    
    def convert_split(self, dataset: Dataset, split_name: str) -> None:
        """Convert a dataset split to disk format."""
        split_dir = self.output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Converting {split_name} split...")
        
        # Create progress bar
        progress_bar = tqdm(
            enumerate(dataset),
            desc=f"Converting {split_name:5}",
            total=len(dataset),
            unit="samples",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        successful = 0
        failed = 0
        
        for idx, example in progress_bar:
            example_id = f"{split_name}_{idx:06d}"
            
            if self.process_example(example, example_id, split_dir):
                successful += 1
            else:
                failed += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Success': f'{successful:,}',
                'Failed': f'{failed:,}',
                'Rate': f'{successful/(successful+failed):.2%}' if (successful+failed) > 0 else '0%'
            })
        
        progress_bar.close()
        
        # Update statistics
        self.stats['splits'][split_name] = successful
        self.stats['successful_conversions'] += successful
        self.stats['failed_conversions'] += failed
        
        print(f"✅ {split_name} conversion completed: {successful:,} successful, {failed:,} failed")
    
    def convert_dataset(self) -> None:
        """Convert the entire dataset."""
        print("🚀 Starting dataset conversion...")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🎯 Target format: screenshot.png + structure.json + layout.json")
        print("-" * 60)
        
        # Load dataset
        dataset = self.load_dataset()
        
        # Validate format
        if not self.validate_dataset_format(dataset):
            return
        
        # Apply column mapping if specified
        if self.args.column_mapping:
            dataset = self.apply_column_mapping(dataset)
        
        # Create splits
        train_dataset, val_dataset, test_dataset = self.create_splits(dataset)
        
        # Convert each split
        self.convert_split(train_dataset, 'train')
        self.convert_split(val_dataset, 'val')
        self.convert_split(test_dataset, 'test')
        
        # Save conversion metadata
        self.save_conversion_metadata()
        
        # Print final statistics
        self.print_final_statistics()
    
    def apply_column_mapping(self, dataset: Dataset) -> Dataset:
        """Apply column name mappings."""
        mappings = {}
        for mapping in self.args.column_mapping:
            old_name, new_name = mapping.split(':')
            mappings[old_name] = new_name
        
        print(f"🔄 Applying column mappings: {mappings}")
        
        # Rename columns
        for old_name, new_name in mappings.items():
            if old_name in dataset.column_names:
                dataset = dataset.rename_column(old_name, new_name)
        
        return dataset
    
    def save_conversion_metadata(self) -> None:
        """Save metadata about the conversion process."""
        metadata = {
            'conversion_timestamp': datetime.now().isoformat(),
            'source_dataset': self.args.dataset_name or self.args.dataset_path,
            'output_directory': str(self.output_dir),
            'split_ratios': {
                'train': self.train_ratio,
                'val': self.val_ratio,
                'test': self.test_ratio
            },
            'statistics': self.stats,
            'conversion_args': vars(self.args)
        }
        
        metadata_path = self.output_dir / 'conversion_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Conversion metadata saved to: {metadata_path}")
    
    def print_final_statistics(self) -> None:
        """Print final conversion statistics."""
        print("\n" + "="*60)
        print("🏁 DATASET CONVERSION COMPLETED!")
        print("="*60)
        
        print(f"📊 Conversion Statistics:")
        print(f"   Total Processed: {self.stats['successful_conversions'] + self.stats['failed_conversions']:,}")
        print(f"   Successful: {self.stats['successful_conversions']:,}")
        print(f"   Failed: {self.stats['failed_conversions']:,}")
        print(f"   Success Rate: {self.stats['successful_conversions']/(self.stats['successful_conversions']+self.stats['failed_conversions']):.2%}")
        
        print(f"\n📁 Dataset Splits:")
        print(f"   Train: {self.stats['splits']['train']:,} samples")
        print(f"   Val:   {self.stats['splits']['val']:,} samples")
        print(f"   Test:  {self.stats['splits']['test']:,} samples")
        print(f"   Total: {sum(self.stats['splits'].values()):,} samples")
        
        print(f"\n📂 Output Structure:")
        print(f"   {self.output_dir}/")
        print(f"   ├── train/          ({self.stats['splits']['train']:,} examples)")
        print(f"   ├── val/            ({self.stats['splits']['val']:,} examples)")
        print(f"   ├── test/           ({self.stats['splits']['test']:,} examples)")
        print(f"   └── conversion_metadata.json")
        
        print(f"\n🚀 Ready for Training!")
        print(f"   Use: python3 scripts/train_model.py --dataset_dir {self.output_dir} --auto_phase")
        
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Convert Hugging Face dataset to training format')
    
    # Source dataset options
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--dataset_name', type=str,
                            help='Hugging Face dataset name (e.g., "username/dataset-name")')
    source_group.add_argument('--dataset_path', type=str,
                            help='Path to local dataset directory or file')
    
    # Dataset options
    parser.add_argument('--dataset_split', type=str, default='train',
                        help='Which split to use from HuggingFace dataset (default: train)')
    parser.add_argument('--streaming', action='store_true',
                        help='Use streaming mode for large datasets')
    
    # Output options
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for processed dataset')
    parser.add_argument('--num_samples', type=int,
                        help='Limit number of samples to process')
    parser.add_argument('--split_ratio', type=str, default='0.8,0.1,0.1',
                        help='Train/val/test split ratios (default: 0.8,0.1,0.1)')
    
    # Column mapping options
    parser.add_argument('--column_mapping', nargs='*',
                        help='Column name mappings in format old:new (e.g., image:screenshot)')
    
    # Processing options
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output directory')
    parser.add_argument('--image_format', choices=['png', 'jpg'], default='png',
                        help='Output image format (default: png)')
    parser.add_argument('--image_size', type=int,
                        help='Resize images to specified size (maintains aspect ratio)')
    
    # Random seed for reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible splits (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Validate arguments
    if args.overwrite and Path(args.output_dir).exists():
        import shutil
        print(f"🗑️ Removing existing output directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    elif Path(args.output_dir).exists() and any(Path(args.output_dir).iterdir()):
        print(f"❌ Output directory {args.output_dir} exists and is not empty.")
        print("Use --overwrite to replace existing data.")
        sys.exit(1)
    
    # Create converter and run
    try:
        converter = DatasetConverter(args)
        converter.convert_dataset()
    except KeyboardInterrupt:
        print("\n⚠️ Conversion interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 