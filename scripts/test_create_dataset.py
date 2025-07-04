#!/usr/bin/env python3
"""
Test script for create_dataset.py

This script creates a small sample dataset and tests the conversion functionality.
"""

import json
import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def create_sample_dataset():
    """Create a small sample dataset for testing."""
    print("🧪 Creating sample dataset for testing...")
    
    # Create sample directory
    sample_dir = Path("data/sample_test")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample data
    samples = []
    for i in range(5):
        # Create a simple test image
        img = Image.new('RGB', (256, 256), color=(i*50, 100, 150))
        
        # Create sample structure
        structure = {
            "div.container": {
                f"h1.title_{i}": {"text": f"Sample Title {i}"},
                "div.content": {
                    "p.text": {"text": f"Sample content for item {i}"}
                }
            }
        }
        
        # Create sample layout
        layout = {
            "structure": {
                f"section@div.container": {
                    f"heading@h1.title_{i}": "",
                    "content@div.content": {
                        "text@p.text": ""
                    }
                }
            },
            "props": {
                "bi": "div.background_image"
            }
        }
        
        samples.append({
            "image": img,
            "structure": structure,
            "layout": layout
        })
    
    # Save as JSON dataset
    dataset_file = sample_dir / "test_dataset.json"
    
    # Convert to JSON-serializable format
    json_samples = []
    for i, sample in enumerate(samples):
        # Save image
        img_path = sample_dir / f"image_{i}.png"
        sample["image"].save(img_path)
        
        json_samples.append({
            "image": str(img_path),
            "structure": json.dumps(sample["structure"]),
            "layout": json.dumps(sample["layout"])
        })
    
    # Save dataset JSON
    with open(dataset_file, 'w') as f:
        json.dump(json_samples, f, indent=2)
    
    print(f"✅ Sample dataset created: {dataset_file}")
    print(f"📊 Contains {len(samples)} examples")
    return dataset_file

def test_conversion():
    """Test the dataset conversion."""
    print("\n🔄 Testing dataset conversion...")
    
    # Create sample dataset
    dataset_file = create_sample_dataset()
    
    # Test the conversion
    import subprocess
    
    cmd = [
        sys.executable, "scripts/create_dataset.py",
        "--dataset_path", str(dataset_file),
        "--output_dir", "data/test_output",
        "--num_samples", "3",
        "--split_ratio", "0.6,0.2,0.2",
        "--overwrite"
    ]
    
    print(f"🚀 Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Conversion successful!")
            print("📁 Output structure:")
            
            output_dir = Path("data/test_output")
            if output_dir.exists():
                for split in ['train', 'val', 'test']:
                    split_dir = output_dir / split
                    if split_dir.exists():
                        count = len(list(split_dir.iterdir()))
                        print(f"   {split}/: {count} examples")
            
            return True
        else:
            print("❌ Conversion failed!")
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running conversion: {e}")
        return False

def main():
    print("🧪 TESTING CREATE_DATASET.PY SCRIPT")
    print("=" * 50)
    
    # Test conversion
    success = test_conversion()
    
    if success:
        print("\n✅ All tests passed!")
        print("\n📝 Usage examples:")
        print("   # From Hugging Face:")
        print("   python3 scripts/create_dataset.py --dataset_name 'username/dataset' --output_dir data/raw")
        print("\n   # From local file:")
        print("   python3 scripts/create_dataset.py --dataset_path dataset.json --output_dir data/raw")
    else:
        print("\n❌ Tests failed!")
        print("💡 Check the error messages above for troubleshooting")

if __name__ == "__main__":
    main() 