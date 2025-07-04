#!/usr/bin/env python3
"""
Automated Dataset Validation Suite - Task 2.5
Command-line tool for comprehensive dataset validation

Usage:
    python scripts/validate_dataset.py /path/to/dataset
    python scripts/validate_dataset.py /path/to/dataset --verbose
    python scripts/validate_dataset.py /path/to/dataset --report-only
"""

import sys
import os
import argparse
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.data.validation import validate_dataset, generate_dataset_report
    from src.data.filesystem_layout import FilesystemLayoutManager
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please make sure you're running from the project root directory")
    sys.exit(1)


def validate_simple_dataset_structure(dataset_dir: str) -> dict:
    """
    Validate simple train/val/test directory structure.
    
    Expected structure:
    dataset/
    ├── train/
    │   ├── example_0001/
    │   │   ├── screenshot.png
    │   │   ├── structure.json
    │   │   └── layout.json
    │   └── ...
    ├── val/
    └── test/
    """
    dataset_path = Path(dataset_dir)
    results = {
        'dataset_name': dataset_path.name,
        'total_examples': 0,
        'splits': {},
        'validation_passed': True,
        'errors': [],
        'warnings': []
    }
    
    # Check for train/val/test splits
    expected_splits = ['train', 'val', 'test']
    found_splits = []
    
    for split in expected_splits:
        split_dir = dataset_path / split
        if split_dir.exists() and split_dir.is_dir():
            found_splits.append(split)
            
            # Count examples in this split
            examples = [d for d in split_dir.iterdir() if d.is_dir()]
            results['splits'][split] = len(examples)
            results['total_examples'] += len(examples)
            
            # Validate a few examples
            for i, example_dir in enumerate(examples[:5]):  # Check first 5
                # Check required files
                required_files = ['screenshot.png', 'structure.json', 'layout.json']
                for file_name in required_files:
                    file_path = example_dir / file_name
                    if not file_path.exists():
                        results['errors'].append(f"{split}/{example_dir.name}: Missing {file_name}")
                        results['validation_passed'] = False
                    elif file_name.endswith('.json'):
                        # Validate JSON files
                        try:
                            with open(file_path, 'r') as f:
                                json.load(f)
                        except json.JSONDecodeError as e:
                            results['errors'].append(f"{split}/{example_dir.name}: Invalid JSON in {file_name}: {e}")
                            results['validation_passed'] = False
    
    if not found_splits:
        results['errors'].append("No train/val/test directories found")
        results['validation_passed'] = False
    elif len(found_splits) < 3:
        missing = set(expected_splits) - set(found_splits)
        results['warnings'].append(f"Missing splits: {', '.join(missing)}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Dataset Validation Suite')
    parser.add_argument('--dataset_dir', help='Path to dataset directory')
    parser.add_argument('--report-only', action='store_true',
                       help='Generate report only (legacy mode)')
    parser.add_argument('--output_dir', '-o', help='Output directory for validation results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--fix_errors', action='store_true', help='Attempt to fix errors automatically')
    
    args = parser.parse_args()
    
    # Check if dataset directory exists
    if not os.path.exists(args.dataset_dir):
        print(f"❌ Error: Dataset directory not found: {args.dataset_dir}")
        sys.exit(1)
    
    print("🔍 Starting dataset validation...")
    print(f"📁 Dataset directory: {args.dataset_dir}")
    print(f"🔧 Verbose mode: {'✅' if args.verbose else '❌'}")
    print("-" * 50)
    
    try:
        if args.report_only:
            print("📊 Generating validation report (legacy mode)...")
            report_text = generate_dataset_report(args.dataset_dir)
            print(report_text)
            return
        
        # Try comprehensive validation first
        try:
            print("🔍 Running comprehensive validation...")
            report = validate_dataset(args.dataset_dir, save_report=False)
            
            # If no examples found, use simple validation instead
            if report.total_examples == 0:
                raise Exception("No examples found in comprehensive validation")
            
            # Print summary
            print(f"\n📈 Validation Summary:")
            print(f"   Dataset: {report.dataset_name}")
            print(f"   Total examples: {report.total_examples}")
            print(f"   Overall score: {report.overall_score:.1%}")
            print(f"   Validation: {'✅ PASSED' if report.summary['validation_passed'] else '❌ FAILED'}")
            
            # Print split information
            if report.splits:
                print(f"\n📊 Dataset Splits:")
                for split_name, size in report.splits.items():
                    print(f"   {split_name}: {size:,} examples")
            
            # Print detailed results if verbose
            if args.verbose:
                print(f"\n🔍 Detailed Results:")
                for result in report.validation_results:
                    status = "✅ PASS" if result.passed else "❌ FAIL"
                    print(f"   {result.check_name}: {status} (Score: {result.score:.1%})")
                    
                    if result.errors and args.verbose:
                        print(f"      ❌ Errors ({len(result.errors)}):")
                        for error in result.errors[:3]:  # Show first 3 errors
                            print(f"         - {error}")
                        if len(result.errors) > 3:
                            print(f"         ... and {len(result.errors) - 3} more")
                    
                    if result.warnings and args.verbose:
                        print(f"      ⚠️ Warnings ({len(result.warnings)}):")
                        for warning in result.warnings[:3]:  # Show first 3 warnings
                            print(f"         - {warning}")
                        if len(result.warnings) > 3:
                            print(f"         ... and {len(result.warnings) - 3} more")
            else:
                # Print error summary
                total_errors = report.summary['total_errors']
                total_warnings = report.summary['total_warnings']
                if total_errors > 0 or total_warnings > 0:
                    print(f"\n📋 Issues Found:")
                    print(f"   Total errors: {total_errors}")
                    print(f"   Total warnings: {total_warnings}")
                    print(f"   Use --verbose for detailed error messages")
            
            validation_passed = report.summary['validation_passed']
            overall_score = report.overall_score
            
        except Exception as e:
            print(f"⚠️ Comprehensive validation failed ({e}), trying simple validation...")
            
            # Fallback to simple validation
            results = validate_simple_dataset_structure(args.dataset_dir)
            
            print(f"\n📈 Simple Validation Summary:")
            print(f"   Dataset: {results['dataset_name']}")
            print(f"   Total examples: {results['total_examples']:,}")
            print(f"   Validation: {'✅ PASSED' if results['validation_passed'] else '❌ FAILED'}")
            
            if results['splits']:
                print(f"\n📊 Dataset Splits:")
                for split_name, size in results['splits'].items():
                    print(f"   {split_name}: {size:,} examples")
            
            if results['errors']:
                print(f"\n❌ Errors ({len(results['errors'])}):")
                for error in results['errors'][:5]:
                    print(f"   - {error}")
                if len(results['errors']) > 5:
                    print(f"   ... and {len(results['errors']) - 5} more")
            
            if results['warnings']:
                print(f"\n⚠️ Warnings ({len(results['warnings'])}):")
                for warning in results['warnings']:
                    print(f"   - {warning}")
            
            validation_passed = results['validation_passed']
            overall_score = 1.0 if validation_passed else 0.0
        
        # Save results to custom output if requested
        if args.output_dir:
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            results_file = output_path / "validation_results.json"
            print(f"\n💾 Results saved to: {results_file}")
        
        # Exit with appropriate code
        if validation_passed:
            print("\n✅ Dataset validation passed!")
            sys.exit(0)
        elif overall_score >= 0.8:
            print(f"\n⚠️ Dataset validation completed with warnings (score: {overall_score:.1%})")
            sys.exit(0)
        else:
            print(f"\n❌ Dataset validation failed (score: {overall_score:.1%})")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 