# Automated Validation Suite - Task 2.5

This document describes the comprehensive automated validation suite implemented for the Diffusion Section Transformer project.

## 📋 Overview

The validation suite implements all requirements specified in Task 2.5 of the instruction:

1. **File Existence** - Verifies both screenshot.png and example.json are present
2. **JSON Schema** - Validates against the unified schema with required fields
3. **Vocabulary Consistency** - Ensures all tokens belong to predefined vocabularies
4. **Syntax Correctness** - Validates @ concatenation and colon-delimited props
5. **Auto-exclusion** - Automatically excludes failed examples from dataset manifest

## 🏗️ Architecture

### Core Components

#### `DatasetValidator` Class

- **Purpose**: Main validation orchestrator
- **Features**: Comprehensive validation, statistics tracking, auto-exclusion
- **Usage**: Both programmatic and command-line interfaces

#### Validation Checks

1. **File Existence Check**: Ensures required files exist
2. **JSON Schema Validation**: Validates structure against unified schema
3. **Vocabulary Consistency**: Checks token validity using loaders
4. **Syntax Validation**: Validates @ concatenation and props syntax

#### Statistical Analysis

- **Success Rate**: Percentage of valid examples
- **Error Breakdown**: Categorized error counts
- **Vocabulary Coverage**: Usage statistics for structure/layout vocabularies
- **Recommendations**: Automated suggestions based on validation results

## 🔧 Usage

### Command-Line Interface

```bash
# Basic validation
python scripts/validate_dataset.py /path/to/dataset

# Validation with detailed output
python scripts/validate_dataset.py /path/to/dataset --verbose

# Auto-exclude failed examples
python scripts/validate_dataset.py /path/to/dataset --auto-exclude

# Save results to JSON file
python scripts/validate_dataset.py /path/to/dataset --output results.json

# Legacy report mode
python scripts/validate_dataset.py /path/to/dataset --report-only
```

### Programmatic Interface

```python
from src.data.validation import DatasetValidator, generate_dataset_report

# Initialize validator
validator = DatasetValidator(
    dataset_root="/path/to/dataset",
    auto_exclude=True  # Optional: auto-exclude failed examples
)

# Run comprehensive validation
results = validator.validate_dataset()

# Access validation results
validation_results = results['validation_results']
statistics = results['statistics']
recommendations = results['recommendations']

# Print statistics
print(f"Total examples: {statistics['total_examples']}")
print(f"Valid examples: {statistics['valid_examples']}")
print(f"Success rate: {statistics['success_rate']:.1%}")

# Check specific error types
print(f"Missing files: {len(validation_results['missing_files'])}")
print(f"Schema errors: {len(validation_results['schema_errors'])}")
print(f"Vocabulary errors: {len(validation_results['vocab_errors'])}")
print(f"Syntax errors: {len(validation_results['syntax_errors'])}")

# Generate comprehensive report
report = generate_dataset_report("/path/to/dataset")
print(report)
```

## 📊 Validation Results Structure

### Main Results Dictionary

```python
{
    'validation_results': {
        'valid_examples': [...],        # List of valid example paths
        'missing_files': [...],         # List of missing file errors
        'invalid_json': [...],          # List of JSON parsing errors
        'schema_errors': [...],         # List of schema validation errors
        'vocab_errors': [...],          # List of vocabulary errors
        'syntax_errors': [...],         # List of syntax errors
        'excluded_examples': [...]      # List of auto-excluded examples
    },
    'statistics': {
        'total_examples': int,          # Total number of examples processed
        'valid_examples': int,          # Number of valid examples
        'success_rate': float,          # Success rate (0.0 - 1.0)
        'total_errors': int,            # Total number of errors
        'error_count': {...},           # Breakdown of errors by type
        'structure_vocab_coverage': float,  # Structure vocabulary coverage
        'layout_vocab_coverage': float,     # Layout vocabulary coverage
        'vocabulary_coverage': {...}    # Detailed vocabulary usage
    },
    'recommendations': [...]            # List of improvement recommendations
}
```

## 🔍 Validation Checks Details

### 1. File Existence Check

Verifies that each example contains:

- `screenshot.png` - The input image
- `example.json` - The metadata and labels

**Failure Conditions:**

- Missing screenshot.png file
- Missing example.json file
- Inaccessible files due to permissions

### 2. JSON Schema Validation

Validates `example.json` structure against the unified schema:

```json
{
  "id": "string",
  "screenshot": {
    "path": "string",
    "width": "number",
    "height": "number"
  },
  "structure": {
    "type": "string",
    "data": "object"
  },
  "layout": {
    "type": "string",
    "data": {
      "structure": "object",
      "props": "object"
    }
  }
}
```

**Failure Conditions:**

- Missing required top-level fields
- Invalid screenshot metadata
- Malformed structure or layout data

### 3. Vocabulary Consistency Check

Validates that all tokens can be processed by the data loaders:

- Structure tokens processed by `StructureLoader`
- Layout tokens processed by `LabelLoader`

**Failure Conditions:**

- Empty token sequences (indicates processing failure)
- Exception during token processing
- Malformed token structures

### 4. Syntax Correctness Validation

Validates specific syntax requirements:

#### @ Concatenation Syntax

- Format: `element1@element2@element3`
- No empty parts: `element@` is invalid
- Valid element names before each @

#### Props Syntax

- Valid prop keys: `bi` (background image), `bv` (background video), `bo` (background overlay)
- Non-empty prop values
- String format for prop values

**Failure Conditions:**

- Invalid @ concatenation format
- Empty parts in concatenated strings
- Invalid prop keys
- Empty or non-string prop values

## 🚫 Auto-Exclusion Feature

When enabled (`auto_exclude=True`), the validator will:

1. **Create Backup**: Saves original `dataset_config.yaml` as `.backup`
2. **Identify Failed Examples**: Determines which examples failed validation
3. **Update Configuration**: Removes failed examples from dataset splits
4. **Log Changes**: Reports how many examples were excluded per split

### Safety Features

- Always creates backup before modification
- Logs all exclusion actions
- Preserves original dataset structure
- Can be reversed using backup file

## 📈 Statistics and Reporting

### Success Rate Calculation

```python
success_rate = valid_examples / total_examples
```

### Vocabulary Coverage

```python
structure_coverage = used_structure_tokens / total_structure_vocab
layout_coverage = used_layout_tokens / total_layout_vocab
```

### Automated Recommendations

The system provides recommendations based on validation results:

- **Low success rate** → Review data collection process
- **Missing files** → Ensure complete dataset structure
- **Vocabulary errors** → Expand vocabulary or review preprocessing
- **Low vocabulary coverage** → Optimize vocabulary usage

## 🧪 Testing

Comprehensive test suite in `tests/test_validation_suite.py`:

```bash
python tests/test_validation_suite.py
```

### Test Coverage

- ✅ DatasetValidator initialization
- ✅ Comprehensive validation workflow
- ✅ Auto-exclusion functionality
- ✅ Vocabulary consistency checking
- ✅ Report generation
- ✅ Edge cases and error handling

## 🔧 Configuration

The validation suite uses the existing data loader configurations and can be customized through:

1. **Vocabulary Files**: Custom vocabulary definitions
2. **Schema Updates**: Modifications to required fields
3. **Syntax Rules**: Custom validation rules for specific formats

## 🚀 Integration

### Dataset Pipeline Integration

```python
from src.data import DatasetLoader, DatasetValidator

# Validate before loading
validator = DatasetValidator("path/to/dataset")
results = validator.validate_dataset()

if results['statistics']['success_rate'] > 0.95:
    # Proceed with training
    dataset = DatasetLoader("path/to/dataset", split="train")
else:
    # Handle validation failures
    print("Dataset validation failed, check results for details")
```

### CI/CD Integration

```bash
# Exit with error code if validation fails
python scripts/validate_dataset.py /path/to/dataset || exit 1
```

## 📝 Best Practices

1. **Always Run Validation** before training or evaluation
2. **Review Recommendations** to improve dataset quality
3. **Use Auto-exclusion Carefully** - always check backup files
4. **Monitor Vocabulary Coverage** to ensure sufficient diversity
5. **Regular Validation** as dataset grows or changes

## 🔗 Related Documentation

- [Data Loader Usage](../DATA_LOADER_USAGE.md)
- [Main README](../README.md)
- [Project Instructions](../instruction.md)

## 🐛 Troubleshooting

### Common Issues

1. **No examples found**

   - Check dataset_config.yaml exists and is properly formatted
   - Verify split names match directory structure

2. **All examples failing vocabulary check**

   - Check that data loaders can access required vocabulary files
   - Verify JSON structure matches expected format

3. **Auto-exclusion excluding all examples**

   - Review validation criteria
   - Check if dataset structure matches expected schema

4. **Permission errors**
   - Ensure read/write access to dataset directory
   - Check file permissions for screenshot.png and example.json files
