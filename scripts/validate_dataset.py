#!/usr/bin/env python3
"""
Automated Dataset Validation Suite - Task 2.5
Command-line tool for comprehensive dataset validation

Usage:
    python scripts/validate_dataset.py /path/to/dataset
    python scripts/validate_dataset.py /path/to/dataset --auto-exclude
    python scripts/validate_dataset.py /path/to/dataset --report-only
"""

import sys
import os
import argparse
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.validation import DatasetValidator, generate_dataset_report


def main():
    parser = argparse.ArgumentParser(description='Dataset Validation Suite - Task 2.5')
    parser.add_argument('dataset_root', help='Path to dataset root directory')
    parser.add_argument('--auto-exclude', action='store_true', 
                       help='Automatically exclude failed examples from manifest')
    parser.add_argument('--report-only', action='store_true',
                       help='Generate report only (legacy mode)')
    parser.add_argument('--output', '-o', help='Output file for validation results (JSON)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Check if dataset root exists
    if not os.path.exists(args.dataset_root):
        print(f"❌ Error: Dataset root not found: {args.dataset_root}")
        sys.exit(1)
    
    # Set up validator
    if args.report_only:
        print("📊 Generating validation report (legacy mode)...")
        report = generate_dataset_report(args.dataset_root)
        print(report)
        return
    
    # Run comprehensive validation
    print("🔍 Starting comprehensive dataset validation...")
    print(f"📁 Dataset root: {args.dataset_root}")
    print(f"🔧 Auto-exclude: {'✅' if args.auto_exclude else '❌'}")
    print("-" * 50)
    
    validator = DatasetValidator(args.dataset_root, auto_exclude=args.auto_exclude)
    results = validator.validate_dataset()
    
    # Print summary
    stats = results['statistics']
    validation_results = results['validation_results']
    
    print("\n📈 Validation Summary:")
    print(f"   Total examples: {stats['total_examples']}")
    print(f"   Valid examples: {stats['valid_examples']}")
    print(f"   Success rate: {stats['success_rate']:.1%}")
    print(f"   Total errors: {stats['total_errors']}")
    
    # Print error breakdown
    if stats['total_errors'] > 0:
        print("\n❌ Error Breakdown:")
        for error_type, count in stats['error_count'].items():
            if count > 0:
                print(f"   {error_type.replace('_', ' ').title()}: {count}")
    
    # Print vocabulary coverage
    print(f"\n📚 Vocabulary Coverage:")
    print(f"   Structure: {stats['structure_vocab_coverage']:.1%}")
    print(f"   Layout: {stats['layout_vocab_coverage']:.1%}")
    
    # Print exclusions if any
    if validation_results['excluded_examples']:
        print(f"\n🚫 Excluded Examples: {len(validation_results['excluded_examples'])}")
        if args.verbose:
            for example in validation_results['excluded_examples'][:5]:
                print(f"   ❌ {example}")
            if len(validation_results['excluded_examples']) > 5:
                print(f"   ... and {len(validation_results['excluded_examples']) - 5} more")
    
    # Print recommendations
    recommendations = results['recommendations']
    if recommendations:
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    
    # Save results to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {args.output}")
    
    # Exit with appropriate code
    success_rate = stats['success_rate']
    if success_rate == 1.0:
        print("\n✅ Dataset validation passed!")
        sys.exit(0)
    elif success_rate >= 0.9:
        print(f"\n⚠️  Dataset validation completed with warnings (success rate: {success_rate:.1%})")
        sys.exit(0)
    else:
        print(f"\n❌ Dataset validation failed (success rate: {success_rate:.1%})")
        sys.exit(1)


if __name__ == "__main__":
    main() 