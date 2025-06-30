"""
Automated Dataset Validation Suite
Implements comprehensive validation checks as specified in instruction.md
"""

import torch
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from PIL import Image

from .schema import UnifiedExample, UnifiedSchemaValidator
from .filesystem_layout import FilesystemLayoutManager, DatasetManifest
from .data_loaders import VisionLoader, StructuralLoader, LabelLoader


@dataclass
class ValidationResult:
    """Result of a validation check."""
    check_name: str
    passed: bool
    score: float  # 0.0 to 1.0
    details: Dict[str, Any]
    errors: List[str]
    warnings: List[str]


@dataclass
class DatasetValidationReport:
    """Complete validation report for the dataset."""
    dataset_name: str
    validation_date: str
    total_examples: int
    splits: Dict[str, int]
    overall_score: float
    validation_results: List[ValidationResult]
    summary: Dict[str, Any]


class SchemaValidationChecker:
    """Validates dataset against unified schema."""
    
    def __init__(self, filesystem_manager: FilesystemLayoutManager):
        self.filesystem_manager = filesystem_manager
    
    def validate_all_examples(self) -> ValidationResult:
        """Validate all examples against the unified schema."""
        errors = []
        warnings = []
        valid_count = 0
        total_count = 0
        
        example_ids = self.filesystem_manager.list_examples()
        
        for example_id in example_ids:
            total_count += 1
            
            try:
                example_path = self.filesystem_manager.get_example_path(example_id)
                example, example_errors = UnifiedSchemaValidator.load_and_validate_example(example_path)
                
                if example:
                    valid_count += 1
                else:
                    errors.extend([f"{example_id}: {error}" for error in example_errors])
                    
            except Exception as e:
                errors.append(f"{example_id}: Unexpected error - {e}")
        
        score = valid_count / total_count if total_count > 0 else 0.0
        
        return ValidationResult(
            check_name="Schema Validation",
            passed=(score >= 0.95),  # 95% pass rate required
            score=score,
            details={
                "valid_examples": valid_count,
                "total_examples": total_count,
                "invalid_examples": total_count - valid_count
            },
            errors=errors,
            warnings=warnings
        )


class FileIntegrityChecker:
    """Validates file integrity and existence."""
    
    def __init__(self, filesystem_manager: FilesystemLayoutManager):
        self.filesystem_manager = filesystem_manager
    
    def validate_file_integrity(self) -> ValidationResult:
        """Check that all referenced files exist and are accessible."""
        errors = []
        warnings = []
        valid_files = 0
        total_files = 0
        
        example_ids = self.filesystem_manager.list_examples()
        
        for example_id in example_ids:
            try:
                example_path = self.filesystem_manager.get_example_path(example_id)
                
                # Check example JSON exists
                total_files += 1
                if example_path.exists():
                    valid_files += 1
                else:
                    errors.append(f"Missing example file: {example_path}")
                    continue
                
                # Load example and check screenshot
                with open(example_path, 'r') as f:
                    data = json.load(f)
                
                screenshot_path = self.filesystem_manager.get_screenshot_path(data['screenshot']['path'])
                total_files += 1
                
                if screenshot_path.exists():
                    # Check if image can be loaded
                    try:
                        with Image.open(screenshot_path) as img:
                            width, height = img.size
                            if width != data['screenshot']['width'] or height != data['screenshot']['height']:
                                warnings.append(f"{example_id}: Screenshot dimensions mismatch")
                        valid_files += 1
                    except Exception as e:
                        errors.append(f"{example_id}: Corrupted screenshot - {e}")
                else:
                    errors.append(f"{example_id}: Missing screenshot - {screenshot_path}")
                    
            except Exception as e:
                errors.append(f"{example_id}: File integrity check failed - {e}")
        
        score = valid_files / total_files if total_files > 0 else 0.0
        
        return ValidationResult(
            check_name="File Integrity",
            passed=(score >= 0.98),  # 98% file integrity required
            score=score,
            details={
                "valid_files": valid_files,
                "total_files": total_files,
                "missing_files": total_files - valid_files
            },
            errors=errors,
            warnings=warnings
        )


class DataQualityChecker:
    """Validates data quality and consistency."""
    
    def __init__(self, filesystem_manager: FilesystemLayoutManager):
        self.filesystem_manager = filesystem_manager
        self.vision_loader = VisionLoader()
        self.structural_loader = StructuralLoader()
        self.label_loader = LabelLoader()
    
    def validate_data_quality(self) -> ValidationResult:
        """Comprehensive data quality validation."""
        errors = []
        warnings = []
        quality_scores = []
        
        example_ids = self.filesystem_manager.list_examples()
        
        for example_id in example_ids:
            try:
                example_path = self.filesystem_manager.get_example_path(example_id)
                example, example_errors = UnifiedSchemaValidator.load_and_validate_example(example_path)
                
                if not example:
                    continue
                
                # Check screenshot quality
                screenshot_score = self._check_screenshot_quality(example)
                
                # Check structure quality
                structure_score = self._check_structure_quality(example)
                
                # Check layout quality
                layout_score = self._check_layout_quality(example)
                
                # Composite quality score
                overall_score = (screenshot_score + structure_score + layout_score) / 3
                quality_scores.append(overall_score)
                
                # Flag low-quality examples
                if overall_score < 0.5:
                    warnings.append(f"{example_id}: Low quality score {overall_score:.2f}")
                
            except Exception as e:
                errors.append(f"{example_id}: Quality check failed - {e}")
        
        average_quality = np.mean(quality_scores) if quality_scores else 0.0
        
        return ValidationResult(
            check_name="Data Quality",
            passed=(average_quality >= 0.7),  # 70% average quality required
            score=average_quality,
            details={
                "average_quality": average_quality,
                "quality_distribution": {
                    "high (>0.8)": sum(1 for s in quality_scores if s > 0.8),
                    "medium (0.5-0.8)": sum(1 for s in quality_scores if 0.5 <= s <= 0.8),
                    "low (<0.5)": sum(1 for s in quality_scores if s < 0.5)
                }
            },
            errors=errors,
            warnings=warnings
        )
    
    def _check_screenshot_quality(self, example: UnifiedExample) -> float:
        """Check screenshot quality metrics."""
        try:
            screenshot_path = self.filesystem_manager.get_screenshot_path(example.screenshot.path)
            
            with Image.open(screenshot_path) as img:
                # Check dimensions
                width, height = img.size
                if width < 800 or height < 600:
                    return 0.3  # Too small
                
                # Check aspect ratio
                aspect_ratio = width / height
                if aspect_ratio < 0.5 or aspect_ratio > 3.0:
                    return 0.4  # Unusual aspect ratio
                
                # Check if image is not blank
                img_array = np.array(img.convert('L'))
                if np.std(img_array) < 10:
                    return 0.2  # Too uniform (likely blank)
                
                return 1.0  # Good quality
                
        except Exception:
            return 0.0
    
    def _check_structure_quality(self, example: UnifiedExample) -> float:
        """Check HTML structure quality."""
        try:
            structure_data = example.structure.data
            
            if not structure_data:
                return 0.0
            
            # Check depth and complexity
            depth = self._calculate_structure_depth(structure_data)
            if depth < 2:
                return 0.4  # Too shallow
            if depth > 10:
                return 0.6  # Too deep
            
            # Check for text content
            has_text = self._has_text_content(structure_data)
            if not has_text:
                return 0.5  # No text content
            
            return 1.0  # Good structure
            
        except Exception:
            return 0.0
    
    def _check_layout_quality(self, example: UnifiedExample) -> float:
        """Check layout data quality."""
        try:
            layout_data = example.layout.data
            
            if not layout_data or not layout_data.get('structure'):
                return 0.0
            
            # Check @ concatenation syntax
            structure = layout_data['structure']
            has_concatenation = self._has_proper_concatenation(structure)
            if not has_concatenation:
                return 0.3
            
            # Check element diversity
            elements = self._extract_layout_elements(structure)
            if len(set(elements)) < 2:
                return 0.5  # Not diverse enough
            
            return 1.0  # Good layout
            
        except Exception:
            return 0.0
    
    def _calculate_structure_depth(self, structure: Dict[str, Any], current_depth: int = 0) -> int:
        """Calculate maximum depth of HTML structure."""
        max_depth = current_depth
        
        for key, value in structure.items():
            if isinstance(value, dict):
                depth = self._calculate_structure_depth(value, current_depth + 1)
                max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _has_text_content(self, structure: Dict[str, Any]) -> bool:
        """Check if structure contains text content."""
        for key, value in structure.items():
            if isinstance(value, dict):
                if "text" in value and value["text"].strip():
                    return True
                if self._has_text_content(value):
                    return True
            elif isinstance(value, str) and value.strip():
                return True
        
        return False
    
    def _has_proper_concatenation(self, structure: Dict[str, Any]) -> bool:
        """Check if layout uses proper @ concatenation syntax."""
        for key, value in structure.items():
            if "@" not in key:
                return False
            if isinstance(value, dict):
                if not self._has_proper_concatenation(value):
                    return False
        
        return True
    
    def _extract_layout_elements(self, structure: Dict[str, Any]) -> List[str]:
        """Extract element types from layout structure."""
        elements = []
        
        for key, value in structure.items():
            if "@" in key:
                element_type = key.split("@")[0]
                elements.append(element_type)
            
            if isinstance(value, dict):
                nested_elements = self._extract_layout_elements(value)
                elements.extend(nested_elements)
        
        return elements


class SplitBalanceChecker:
    """Validates dataset split balance and distribution."""
    
    def __init__(self, filesystem_manager: FilesystemLayoutManager):
        self.filesystem_manager = filesystem_manager
    
    def validate_split_balance(self) -> ValidationResult:
        """Check dataset split balance and distribution."""
        errors = []
        warnings = []
        
        manifest = self.filesystem_manager.load_manifest()
        if not manifest:
            return ValidationResult(
                check_name="Split Balance",
                passed=False,
                score=0.0,
                details={},
                errors=["No manifest found"],
                warnings=[]
            )
        
        total_examples = sum(split.size for split in manifest.splits.values())
        
        if total_examples == 0:
            return ValidationResult(
                check_name="Split Balance",
                passed=False,
                score=0.0,
                details={},
                errors=["No examples in dataset"],
                warnings=[]
            )
        
        # Calculate split ratios
        split_ratios = {}
        for name, split in manifest.splits.items():
            split_ratios[name] = split.size / total_examples
        
        # Check balance (typical ratios: train=0.8, val=0.15, test=0.05)
        score = 1.0
        
        if "train" in split_ratios:
            train_ratio = split_ratios["train"]
            if train_ratio < 0.6 or train_ratio > 0.9:
                warnings.append(f"Train split ratio {train_ratio:.2f} is outside optimal range (0.6-0.9)")
                score *= 0.8
        
        if "validation" in split_ratios:
            val_ratio = split_ratios["validation"]
            if val_ratio < 0.05 or val_ratio > 0.25:
                warnings.append(f"Validation split ratio {val_ratio:.2f} is outside optimal range (0.05-0.25)")
                score *= 0.9
        
        if "test" in split_ratios:
            test_ratio = split_ratios["test"]
            if test_ratio < 0.05 or test_ratio > 0.25:
                warnings.append(f"Test split ratio {test_ratio:.2f} is outside optimal range (0.05-0.25)")
                score *= 0.9
        
        # Check minimum split sizes
        for name, split in manifest.splits.items():
            if split.size > 0 and split.size < 10:
                warnings.append(f"{name} split has only {split.size} examples (minimum 10 recommended)")
                score *= 0.8
        
        return ValidationResult(
            check_name="Split Balance",
            passed=(score >= 0.7),
            score=score,
            details={
                "total_examples": total_examples,
                "split_sizes": {name: split.size for name, split in manifest.splits.items()},
                "split_ratios": split_ratios
            },
            errors=errors,
            warnings=warnings
        )


class DatasetValidator:
    """Main dataset validation orchestrator."""
    
    def __init__(self, filesystem_manager: FilesystemLayoutManager):
        self.filesystem_manager = filesystem_manager
        
        # Initialize checkers
        self.schema_checker = SchemaValidationChecker(filesystem_manager)
        self.integrity_checker = FileIntegrityChecker(filesystem_manager)
        self.quality_checker = DataQualityChecker(filesystem_manager)
        self.balance_checker = SplitBalanceChecker(filesystem_manager)
    
    def run_full_validation(self, save_report: bool = True) -> DatasetValidationReport:
        """
        Run complete dataset validation suite.
        
        Args:
            save_report: Whether to save validation report to file
            
        Returns:
            Comprehensive validation report
        """
        print("🔍 Starting comprehensive dataset validation...")
        
        # Run all validation checks
        validation_results = []
        
        print("   Validating schema compliance...")
        schema_result = self.schema_checker.validate_all_examples()
        validation_results.append(schema_result)
        
        print("   Checking file integrity...")
        integrity_result = self.integrity_checker.validate_file_integrity()
        validation_results.append(integrity_result)
        
        print("   Assessing data quality...")
        quality_result = self.quality_checker.validate_data_quality()
        validation_results.append(quality_result)
        
        print("   Validating split balance...")
        balance_result = self.balance_checker.validate_split_balance()
        validation_results.append(balance_result)
        
        # Calculate overall score
        overall_score = np.mean([result.score for result in validation_results])
        
        # Get dataset statistics
        manifest = self.filesystem_manager.load_manifest()
        if manifest:
            total_examples = manifest.metadata.get('total_examples', 0)
            splits = {name: split.size for name, split in manifest.splits.items()}
            dataset_name = manifest.metadata.get('name', 'Unknown')
        else:
            total_examples = 0
            splits = {}
            dataset_name = 'Unknown'
        
        # Create summary
        summary = {
            "validation_passed": all(result.passed for result in validation_results),
            "total_errors": sum(len(result.errors) for result in validation_results),
            "total_warnings": sum(len(result.warnings) for result in validation_results),
            "checks_passed": sum(1 for result in validation_results if result.passed),
            "total_checks": len(validation_results)
        }
        
        # Create report
        report = DatasetValidationReport(
            dataset_name=dataset_name,
            validation_date=datetime.now().isoformat(),
            total_examples=total_examples,
            splits=splits,
            overall_score=overall_score,
            validation_results=validation_results,
            summary=summary
        )
        
        # Save report if requested
        if save_report:
            self._save_validation_report(report)
        
        # Print summary
        self._print_validation_summary(report)
        
        return report
    
    def _save_validation_report(self, report: DatasetValidationReport):
        """Save validation report to file."""
        report_dir = self.filesystem_manager.validation_dir
        report_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"validation_report_{timestamp}.json"
        
        # Convert report to dictionary for JSON serialization
        report_dict = {
            "dataset_name": report.dataset_name,
            "validation_date": report.validation_date,
            "total_examples": report.total_examples,
            "splits": report.splits,
            "overall_score": report.overall_score,
            "summary": report.summary,
            "validation_results": [
                {
                    "check_name": result.check_name,
                    "passed": result.passed,
                    "score": result.score,
                    "details": result.details,
                    "errors": result.errors,
                    "warnings": result.warnings
                }
                for result in report.validation_results
            ]
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        print(f"✅ Validation report saved to: {report_path}")
    
    def _print_validation_summary(self, report: DatasetValidationReport):
        """Print validation summary to console."""
        print("\n" + "="*60)
        print(f"🎯 DATASET VALIDATION REPORT")
        print("="*60)
        print(f"Dataset: {report.dataset_name}")
        print(f"Total Examples: {report.total_examples}")
        print(f"Overall Score: {report.overall_score:.2f}")
        print(f"Validation: {'✅ PASSED' if report.summary['validation_passed'] else '❌ FAILED'}")
        print()
        
        print("📊 Split Distribution:")
        for split_name, size in report.splits.items():
            print(f"   {split_name}: {size} examples")
        print()
        
        print("🔍 Validation Results:")
        for result in report.validation_results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"   {result.check_name}: {status} (Score: {result.score:.2f})")
            
            if result.errors:
                print(f"      ❌ {len(result.errors)} errors")
            if result.warnings:
                print(f"      ⚠️ {len(result.warnings)} warnings")
        
        print()
        print("📈 Summary:")
        print(f"   Checks Passed: {report.summary['checks_passed']}/{report.summary['total_checks']}")
        print(f"   Total Errors: {report.summary['total_errors']}")
        print(f"   Total Warnings: {report.summary['total_warnings']}")
        print("="*60)


def validate_dataset(
    base_data_dir: str = "data",
    save_report: bool = True
) -> DatasetValidationReport:
    """
    Convenience function to validate dataset.
    
    Args:
        base_data_dir: Base directory containing dataset
        save_report: Whether to save validation report
        
    Returns:
        Validation report
    """
    from .filesystem_layout import FilesystemLayoutManager
    
    filesystem_manager = FilesystemLayoutManager(base_data_dir)
    validator = DatasetValidator(filesystem_manager)
    
    return validator.run_full_validation(save_report=save_report)
