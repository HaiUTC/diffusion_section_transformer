"""
Comprehensive Tests for Step 2: Dataset Pipeline
Tests all components of the dataset pipeline as specified in instruction.md
"""

import pytest
import torch
import json
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.schema import (
    UnifiedExample, ScreenshotMetadata, HTMLStructureData, SectionLayoutData,
    UnifiedSchemaValidator, create_example_template, create_example_with_background
)
from src.data.filesystem_layout import (
    FilesystemLayoutManager, DatasetManifest, DatasetSplit, setup_dataset_structure
)
from src.data.data_loaders import (
    VisionLoader, StructuralLoader, LabelLoader, MultimodalLayoutDataset, create_data_loaders
)
from src.data.validation import (
    DatasetValidator, ValidationResult, validate_dataset
)


class TestUnifiedSchema:
    """Test the unified JSON schema implementation."""
    
    def test_example_creation(self):
        """Test creating unified examples."""
        # Test basic template
        template = create_example_template("test_001")
        
        assert template["id"] == "test_001"
        assert template["screenshot"]["path"] == "screenshot.png"
        assert template["structure"]["type"] == "HTMLObject"
        assert template["layout"]["type"] == "SectionLayout"
        assert "@" in list(template["layout"]["data"]["structure"].keys())[0]
    
    def test_background_example_creation(self):
        """Test creating examples with background properties."""
        # Test image background
        bg_template = create_example_with_background("test_bg_001", "image")
        
        assert "bi" in bg_template["layout"]["props"]
        assert "bo" in bg_template["layout"]["props"]
        
        # Test video background
        video_template = create_example_with_background("test_bg_002", "video")
        
        assert "bv" in video_template["layout"]["props"]
        assert "bo" in video_template["layout"]["props"]
    
    def test_schema_validation(self):
        """Test schema validation functionality."""
        # Valid example
        valid_example = create_example_template("test_valid")
        is_valid, errors = UnifiedSchemaValidator.validate_example(valid_example)
        
        assert is_valid
        assert len(errors) == 0
        
        # Invalid example - missing required field
        invalid_example = valid_example.copy()
        del invalid_example["screenshot"]
        
        is_valid, errors = UnifiedSchemaValidator.validate_example(invalid_example)
        assert not is_valid
        assert any("screenshot" in error for error in errors)
        
        # Invalid concatenation syntax
        invalid_concat = valid_example.copy()
        invalid_concat["layout"]["data"]["structure"] = {
            "invalid_key": {}  # Missing @ symbol
        }
        
        is_valid, errors = UnifiedSchemaValidator.validate_example(invalid_concat)
        assert not is_valid
        assert any("concatenation" in error for error in errors)


class TestFilesystemLayout:
    """Test filesystem layout and dataset management."""
    
    def setup_method(self):
        """Setup temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.filesystem_manager = FilesystemLayoutManager(self.temp_dir)
    
    def teardown_method(self):
        """Cleanup temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_directory_structure_creation(self):
        """Test creating the required directory structure."""
        self.filesystem_manager.create_directory_structure()
        
        # Check all required directories exist
        assert self.filesystem_manager.raw_dir.exists()
        assert self.filesystem_manager.processed_dir.exists()
        assert self.filesystem_manager.examples_dir.exists()
        assert self.filesystem_manager.screenshots_dir.exists()
        assert self.filesystem_manager.cache_dir.exists()
        assert self.filesystem_manager.validation_dir.exists()
    
    def test_manifest_creation_and_loading(self):
        """Test dataset manifest creation and loading."""
        self.filesystem_manager.create_directory_structure()
        
        # Create manifest
        manifest = self.filesystem_manager.create_manifest_template()
        self.filesystem_manager.save_manifest(manifest)
        
        # Load manifest
        loaded_manifest = self.filesystem_manager.load_manifest()
        
        assert loaded_manifest is not None
        assert loaded_manifest.metadata["name"] == "section_layout_dataset"
        assert "train" in loaded_manifest.splits
        assert "validation" in loaded_manifest.splits
        assert "test" in loaded_manifest.splits
    
    def test_example_addition(self):
        """Test adding examples to the dataset."""
        self.filesystem_manager.create_directory_structure()
        
        # Create example
        example_data = create_example_template("test_example_001")
        example = UnifiedExample(
            id=example_data["id"],
            screenshot=ScreenshotMetadata(**example_data["screenshot"]),
            structure=HTMLStructureData(**example_data["structure"]),
            layout=SectionLayoutData(**example_data["layout"])
        )
        
        # Add example
        success = self.filesystem_manager.add_example(example, "train")
        assert success
        
        # Verify example was added
        manifest = self.filesystem_manager.load_manifest()
        assert manifest.splits["train"].size == 1
        assert "test_example_001" in manifest.splits["train"].examples
        
        # Verify file exists
        example_path = self.filesystem_manager.get_example_path("test_example_001")
        assert example_path.exists()
    
    def test_split_indices_creation(self):
        """Test creating balanced split indices."""
        splits = self.filesystem_manager.create_split_indices(
            total_examples=1000,
            train_ratio=0.8,
            val_ratio=0.15,
            test_ratio=0.05
        )
        
        assert len(splits["train"]) == 800
        assert len(splits["validation"]) == 150
        assert len(splits["test"]) == 50
        
        # Check no overlap
        all_ids = set()
        for split_ids in splits.values():
            assert len(set(split_ids) & all_ids) == 0  # No duplicates
            all_ids.update(split_ids)


class TestDataLoaders:
    """Test data loading components."""
    
    def setup_method(self):
        """Setup test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.filesystem_manager = FilesystemLayoutManager(self.temp_dir)
        self.filesystem_manager.create_directory_structure()
        
        # Create test screenshot
        self.test_image = Image.new('RGB', (224, 224), color='red')
        self.screenshot_path = self.filesystem_manager.screenshots_dir / "test_screenshot.png"
        self.test_image.save(self.screenshot_path)
    
    def teardown_method(self):
        """Cleanup."""
        shutil.rmtree(self.temp_dir)
    
    def test_vision_loader(self):
        """Test vision data loading."""
        vision_loader = VisionLoader(image_size=224, patch_size=16)
        
        # Test image loading
        image_tensor = vision_loader.load_screenshot(self.screenshot_path)
        
        assert image_tensor.shape == (3, 224, 224)
        assert image_tensor.dtype == torch.float32
        
        # Test patch extraction
        patches = vision_loader.extract_patches(image_tensor)
        expected_num_patches = (224 // 16) ** 2
        expected_patch_dim = 3 * 16 * 16
        
        assert patches.shape == (expected_num_patches, expected_patch_dim)
    
    def test_structural_loader(self):
        """Test structural data loading."""
        structural_loader = StructuralLoader(vocab_size=1000, max_sequence_length=128)
        
        # Test vocabulary building
        structure_data_list = [
            {
                "type": "HTMLObject",
                "data": {
                    "div.container": {
                        "h1.heading": {"text": "Hello World"},
                        "p.paragraph": {"text": "This is a test"}
                    }
                }
            }
        ]
        
        structural_loader.build_vocabulary(structure_data_list)
        
        assert len(structural_loader.token_to_id) > len(structural_loader.special_tokens)
        
        # Test tokenization
        tokens = structural_loader.structure_to_tokens(structure_data_list[0])
        
        assert tokens.shape == (128,)  # max_sequence_length
        assert tokens.dtype == torch.long
        
        # Test attention mask
        mask = structural_loader.create_attention_mask(tokens)
        assert mask.shape == tokens.shape
    
    def test_label_loader(self):
        """Test label data loading."""
        label_loader = LabelLoader(element_vocab_size=50, property_vocab_size=20, max_elements=16)
        
        # Test vocabulary building
        layout_data_list = [
            {
                "type": "SectionLayout",
                "data": {
                    "structure": {
                        "section@div.container": {
                            "heading@h1.title": "",
                            "paragraph@p.content": ""
                        }
                    }
                },
                "props": {"bi": "background_image"}
            }
        ]
        
        label_loader.build_label_vocabulary(layout_data_list)
        
        assert len(label_loader.element_to_id) > len(label_loader.special_element_tokens)
        
        # Test tokenization
        tokens = label_loader.layout_to_tokens(layout_data_list[0])
        
        assert "element_tokens" in tokens
        assert "property_tokens" in tokens
        assert "num_elements" in tokens
        
        assert tokens["element_tokens"].shape == (16,)  # max_elements
        assert tokens["property_tokens"].shape == (20,)  # property_vocab_size
    
    def test_multimodal_dataset(self):
        """Test complete multimodal dataset."""
        # Add sample data
        example_data = create_example_template("test_001", "test_screenshot.png")
        example = UnifiedExample(
            id=example_data["id"],
            screenshot=ScreenshotMetadata(**example_data["screenshot"]),
            structure=HTMLStructureData(**example_data["structure"]),
            layout=SectionLayoutData(**example_data["layout"])
        )
        
        self.filesystem_manager.add_example(example, "train")
        
        # Create dataset
        dataset = MultimodalLayoutDataset(
            self.filesystem_manager,
            split="train",
            image_size=224,
            structure_vocab_size=1000,
            element_vocab_size=50
        )
        
        # Build vocabularies
        dataset.build_vocabularies(["train"])
        
        assert len(dataset) == 1
        
        # Test data loading
        data_item = dataset[0]
        
        assert "screenshot" in data_item
        assert "structure_tokens" in data_item
        assert "structure_mask" in data_item
        assert "element_tokens" in data_item
        assert "property_tokens" in data_item
        assert "num_elements" in data_item
        assert "example_id" in data_item
        
        assert data_item["screenshot"].shape == (3, 224, 224)


class TestDatasetValidation:
    """Test dataset validation suite."""
    
    def setup_method(self):
        """Setup test dataset."""
        self.temp_dir = tempfile.mkdtemp()
        self.filesystem_manager = FilesystemLayoutManager(self.temp_dir)
        self.filesystem_manager.create_directory_structure()
        
        # Create valid test data
        self._create_test_dataset()
    
    def teardown_method(self):
        """Cleanup."""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_dataset(self):
        """Create a small valid test dataset."""
        # Create test screenshots
        for i in range(3):
            image = Image.new('RGB', (800, 600), color=('red', 'green', 'blue')[i])
            screenshot_path = self.filesystem_manager.screenshots_dir / f"test_{i}.png"
            image.save(screenshot_path)
            
            # Create example
            example_data = create_example_template(f"test_{i:03d}", f"test_{i}.png")
            example_data["screenshot"]["width"] = 800
            example_data["screenshot"]["height"] = 600
            
            example = UnifiedExample(
                id=example_data["id"],
                screenshot=ScreenshotMetadata(**example_data["screenshot"]),
                structure=HTMLStructureData(**example_data["structure"]),
                layout=SectionLayoutData(**example_data["layout"])
            )
            
            split = "train" if i < 2 else "validation"
            self.filesystem_manager.add_example(example, split)
    
    def test_schema_validation(self):
        """Test schema validation checker."""
        validator = DatasetValidator(self.filesystem_manager)
        result = validator.validate_schema_compliance()
        
        assert result.passed
        assert result.score == 1.0  # All examples should be valid
        assert result.details["valid_examples"] == 3
        assert len(result.errors) == 0
    
    def test_file_integrity_validation(self):
        """Test file integrity checker."""
        validator = DatasetValidator(self.filesystem_manager)
        result = validator.validate_file_integrity()
        
        assert result.passed
        assert result.score == 1.0  # All files should exist
        assert result.details["valid_files"] == 6  # 3 examples + 3 screenshots
        assert len(result.errors) == 0
    
    def test_data_quality_validation(self):
        """Test data quality checker."""
        validator = DatasetValidator(self.filesystem_manager)
        result = validator.validate_data_quality()
        
        assert result.passed
        assert result.score >= 0.7  # Should meet quality threshold
        assert result.details["examples_checked"] == 3
    
    def test_full_validation_suite(self):
        """Test complete validation suite."""
        validator = DatasetValidator(self.filesystem_manager)
        report = validator.run_full_validation(save_report=False)
        
        assert report.overall_score >= 0.7
        assert report.summary["validation_passed"]
        assert report.summary["total_errors"] == 0
        assert report.total_examples == 3
        
        # Check all validation results
        assert len(report.validation_results) == 3  # Schema, Integrity, Quality
        assert all(result.passed for result in report.validation_results)


class TestIntegrationWorkflow:
    """Integration tests for complete dataset pipeline workflow."""
    
    def setup_method(self):
        """Setup integration test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup."""
        shutil.rmtree(self.temp_dir)
    
    def test_complete_pipeline_workflow(self):
        """Test the complete dataset pipeline from setup to validation."""
        # Step 1: Setup filesystem structure
        filesystem_manager = setup_dataset_structure(
            base_dir=self.temp_dir,
            dataset_name="integration_test_dataset"
        )
        
        assert filesystem_manager.processed_dir.exists()
        assert filesystem_manager.manifest_path.exists()
        
        # Step 2: Create and add examples
        for i in range(10):
            # Create screenshot
            image = Image.new('RGB', (1024, 768), color=np.random.randint(0, 255, 3).tolist())
            screenshot_path = filesystem_manager.screenshots_dir / f"screenshot_{i}.png"
            image.save(screenshot_path)
            
            # Create example with varied data
            example_data = create_example_template(f"example_{i:03d}", f"screenshot_{i}.png")
            example_data["screenshot"]["width"] = 1024
            example_data["screenshot"]["height"] = 768
            
            # Add some variation to structure
            if i % 2 == 0:
                example_data["structure"]["data"]["div.sidebar"] = {
                    "ul.menu": {"text": "Navigation menu"}
                }
            
            # Add background properties occasionally
            if i % 3 == 0:
                example_data = create_example_with_background(f"example_{i:03d}", "image")
                example_data["screenshot"]["path"] = f"screenshot_{i}.png"
                example_data["screenshot"]["width"] = 1024
                example_data["screenshot"]["height"] = 768
            
            example = UnifiedExample(
                id=example_data["id"],
                screenshot=ScreenshotMetadata(**example_data["screenshot"]),
                structure=HTMLStructureData(**example_data["structure"]),
                layout=SectionLayoutData(**example_data["layout"])
            )
            
            # Distribute across splits
            if i < 7:
                split = "train"
            elif i < 9:
                split = "validation"
            else:
                split = "test"
            
            success = filesystem_manager.add_example(example, split)
            assert success
        
        # Step 3: Verify manifest
        manifest = filesystem_manager.load_manifest()
        assert manifest.metadata["total_examples"] == 10
        assert manifest.splits["train"].size == 7
        assert manifest.splits["validation"].size == 2
        assert manifest.splits["test"].size == 1
        
        # Step 4: Create data loaders
        data_loaders = create_data_loaders(
            filesystem_manager,
            batch_size=2,
            num_workers=0,  # No multiprocessing for tests
            image_size=224,
            structure_vocab_size=1000,
            element_vocab_size=50
        )
        
        assert "train" in data_loaders
        assert "validation" in data_loaders
        assert "test" in data_loaders
        
        # Test data loading
        for split_name, data_loader in data_loaders.items():
            for batch in data_loader:
                assert "screenshot" in batch
                assert "structure_tokens" in batch
                assert "element_tokens" in batch
                
                # Check batch dimensions
                batch_size = batch["screenshot"].shape[0]
                assert batch["screenshot"].shape == (batch_size, 3, 224, 224)
                assert batch["structure_tokens"].shape[0] == batch_size
                assert batch["element_tokens"].shape[0] == batch_size
                
                break  # Test just one batch per split
        
        # Step 5: Run validation
        report = validate_dataset(self.temp_dir)
        
        assert report.overall_score >= 0.7
        assert report.summary["validation_passed"]
        assert report.total_examples == 10
        
        print(f"✅ Integration test completed successfully!")
        print(f"   Dataset: {report.dataset_name}")
        print(f"   Examples: {report.total_examples}")
        print(f"   Overall Score: {report.overall_score:.2f}")
        print(f"   Validation: {'PASSED' if report.summary['validation_passed'] else 'FAILED'}")


# Test execution
if __name__ == "__main__":
    # Run key tests
    test_integration = TestIntegrationWorkflow()
    test_integration.setup_method()
    
    try:
        test_integration.test_complete_pipeline_workflow()
        print("🎉 All integration tests passed!")
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        raise
    finally:
        test_integration.teardown_method() 