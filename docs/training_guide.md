# 🚀 Diffusion Section Transformer Training Guide

A comprehensive guide for training your DLT-style model from dataset download to evaluation.

## 📋 Prerequisites

### System Requirements

- **GPU**: NVIDIA GPU with 4GB+ VRAM (8GB+ recommended)
- **RAM**: 16GB+ system memory
- **Storage**: 100GB+ free space
- **Python**: 3.8+
- **CUDA**: 11.0+ (for GPU acceleration)

### Installation

```bash
# Clone and setup project
cd /path/to/your/project
pip install -r requirements.txt

# Verify installation
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

## 🔄 Complete Training Flow

### Phase 1: Dataset Preparation & Download

#### 1.1 Prepare Dataset Structure

```bash
# Create dataset directory structure
mkdir -p data/raw data/processed data/splits
mkdir -p data/raw/{train,val,test}

# Each example should contain:
# - screenshot.png (webpage screenshot)
# - structure.json (HTML object format)
# - layout.json (target section layout)
```

#### 1.2 Dataset Format Specification

**Example Structure** (`data/raw/train/example_0001/`):

```
example_0001/
├── screenshot.png          # 512x512+ webpage screenshot
├── structure.json          # HTML object format
└── layout.json            # Target section layout
```

**structure.json** format:

```json
{
  "div.container": {
    "h1.heading": { "text": "Hello World" },
    "p.paragraph": { "text": "Sample text" },
    "div.grid": {
      "div.column": { "text": "Column 1" },
      "div.column": { "text": "Column 2" }
    }
  }
}
```

**layout.json** format:

```json
{
  "structure": {
    "section@div.container": {
      "heading@h1.heading": "",
      "paragraph@p.paragraph": "",
      "grid@div.grid": {
        "column@div.column": "",
        "column@div.column": ""
      }
    }
  },
  "props": {
    "bi": "div.background_image",
    "bo": "div.background_overlay"
  }
}
```

#### 1.3 Download/Create Dataset

**Option A: Use Existing Dataset**

```bash
# Download your prepared dataset
wget https://your-dataset-url.com/dataset.zip
unzip dataset.zip -d data/raw/
```

**Option B: Create Dataset from Scratch**

```bash
# Use the dataset creation pipeline
python3 scripts/create_dataset.py \
  --source_dir /path/to/webpages \
  --output_dir data/raw \
  --num_samples 5000 \
  --split_ratio 0.8,0.1,0.1
```

#### 1.4 Validate Dataset

```bash
# Validate dataset integrity
python3 scripts/validate_dataset.py \
  --dataset_dir data/raw \
  --fix_errors \
  --verbose
```

Expected output:

```
✅ Found 4,000 training examples
✅ Found 500 validation examples
✅ Found 500 test examples
✅ All examples have required files
✅ JSON schemas are valid
✅ Image dimensions are consistent
```

---

### Phase 2: Training Pipeline

#### 2.1 Automatic Phase Detection

```bash
# The system automatically detects the appropriate training phase
python3 examples/detect_training_phase.py --dataset_dir data/raw

# Example output:
# 📊 Dataset Size: 5,000 samples
# 🎯 Recommended Phase: PHASE2 (Small-Scale Training)
# 💰 Estimated Cost: $100-200/month
# ⏱️ Training Time: 6-12 hours
```

#### 2.2 Start Training

**Quick Start** (Automatic Configuration):

```bash
# Automatic phase detection and training
python3 scripts/train_model.py \
  --dataset_dir data/raw \
  --output_dir models/experiment_1 \
  --auto_phase
```

**Manual Phase Configuration**:

```bash
# Train with specific phase
python3 scripts/train_model.py \
  --dataset_dir data/raw \
  --output_dir models/experiment_1 \
  --phase phase2 \
  --dataset_size 5000 \
  --epochs 100 \
  --batch_size 16
```

**Advanced Configuration**:

```bash
# Custom training with all options
python3 scripts/train_model.py \
  --dataset_dir data/raw \
  --output_dir models/experiment_1 \
  --phase phase2 \
  --dataset_size 5000 \
  --config_file configs/phase2_custom.yaml \
  --resume_from models/checkpoint_epoch_50.pth \
  --distributed \
  --num_gpus 2
```

#### 2.3 Training Phases Overview

| Phase       | Dataset Size | Training Time | Cost/Month | GPU Requirements |
| ----------- | ------------ | ------------- | ---------- | ---------------- |
| **Phase 1** | 0-2K         | 2-4 hours     | $20-50     | 1x T4 (4GB)      |
| **Phase 2** | 2.5K-10K     | 6-12 hours    | $100-200   | 1x V100 (16GB)   |
| **Phase 3** | 25K-100K     | 1-3 days      | $300-500   | 2-4x A100 (40GB) |
| **Phase 4** | 100K+        | 3-7 days      | $800-1500  | 4-8x A100 (80GB) |

#### 2.4 Monitor Training Progress

**Real-time Monitoring**:

```bash
# Monitor training in real-time
tensorboard --logdir models/experiment_1/logs --port 6006

# Or use built-in monitoring
python3 scripts/monitor_training.py --experiment_dir models/experiment_1
```

**Training Metrics to Watch**:

- **Layout Accuracy**: Target >85% for production
- **Element Precision**: Target >80% for production
- **Visual Similarity**: Target >75% for production
- **Loss Convergence**: Should steadily decrease
- **GPU Utilization**: Should be >70% for efficiency

---

### Phase 3: Evaluation & Validation

#### 3.1 Basic Evaluation

```bash
# Evaluate trained model
python3 scripts/evaluate_model.py \
  --model_path models/experiment_1/best_model.pth \
  --test_dir data/raw/test \
  --output_dir results/evaluation_1 \
  --batch_size 32
```

Expected output:

```
📊 EVALUATION RESULTS
====================================
Layout Accuracy:      87.3%
Element Precision:     83.1%
Element Recall:        81.7%
Visual Similarity:     78.9%
Aesthetic Score:       72.4%
Inference Speed:       0.23s/sample
====================================
✅ Model meets production thresholds
```

#### 3.2 Comprehensive Evaluation

```bash
# Detailed evaluation with visualizations
python3 scripts/comprehensive_evaluation.py \
  --model_path models/experiment_1/best_model.pth \
  --test_dir data/raw/test \
  --output_dir results/comprehensive_eval \
  --generate_visualizations \
  --error_analysis \
  --benchmark_comparison
```

Generates:

- **Performance Report**: `results/comprehensive_eval/report.html`
- **Error Analysis**: `results/comprehensive_eval/error_analysis.json`
- **Visualizations**: `results/comprehensive_eval/visualizations/`
- **Benchmark Comparison**: vs. baseline models

#### 3.3 Production Readiness Check

```bash
# Check if model is ready for production deployment
python3 scripts/production_readiness.py \
  --model_path models/experiment_1/best_model.pth \
  --requirements production_requirements.yaml
```

Checks:

- ✅ **Accuracy Thresholds**: >85% layout accuracy
- ✅ **Performance**: <0.5s inference time
- ✅ **Robustness**: Handles edge cases
- ✅ **Memory Usage**: <8GB VRAM for inference
- ✅ **API Compatibility**: RESTful interface ready

---

## 🔧 Advanced Training Workflows

### Distributed Training (Phase 3-4)

```bash
# Multi-GPU distributed training
python3 -m torch.distributed.launch \
  --nproc_per_node=4 \
  --master_port=12355 \
  scripts/train_distributed.py \
  --dataset_dir data/raw \
  --output_dir models/distributed_experiment \
  --phase phase4 \
  --world_size 4
```

### Hyperparameter Optimization

```bash
# Automated hyperparameter tuning
python3 scripts/hyperparameter_optimization.py \
  --dataset_dir data/raw \
  --output_dir models/hpo_experiment \
  --phase phase2 \
  --optimization_method bayesian \
  --num_trials 50 \
  --parallel_trials 4
```

### Curriculum Learning (Phase 2)

```bash
# Curriculum learning with staged training
python3 scripts/curriculum_training.py \
  --dataset_dir data/raw \
  --output_dir models/curriculum_experiment \
  --stage1_epochs 30 \  # Simple layouts
  --stage2_epochs 40 \  # Medium complexity
  --stage3_epochs 30 \  # Complex layouts
  --difficulty_metric element_count
```

---

## 📊 Complete Example Workflow

### Full Training Pipeline Example

```bash
#!/bin/bash
# complete_training_workflow.sh

echo "🚀 Starting Complete Training Workflow"

# Step 1: Prepare dataset
echo "📁 Step 1: Dataset Preparation"
python3 scripts/prepare_dataset.py \
  --source_dir raw_webpages/ \
  --output_dir data/processed \
  --num_samples 8000 \
  --augmentation_factor 10

# Step 2: Validate dataset
echo "✅ Step 2: Dataset Validation"
python3 scripts/validate_dataset.py \
  --dataset_dir data/processed \
  --fix_errors

# Step 3: Detect optimal phase
echo "🎯 Step 3: Phase Detection"
PHASE=$(python3 scripts/detect_phase.py --dataset_dir data/processed --output phase)
echo "Detected phase: $PHASE"

# Step 4: Train model
echo "🏋️ Step 4: Model Training"
python3 scripts/train_model.py \
  --dataset_dir data/processed \
  --output_dir models/production_model \
  --phase $PHASE \
  --auto_config \
  --save_checkpoints \
  --early_stopping

# Step 5: Evaluate model
echo "📊 Step 5: Model Evaluation"
python3 scripts/comprehensive_evaluation.py \
  --model_path models/production_model/best_model.pth \
  --test_dir data/processed/test \
  --output_dir results/final_evaluation

# Step 6: Production readiness check
echo "🚀 Step 6: Production Readiness"
python3 scripts/production_readiness.py \
  --model_path models/production_model/best_model.pth \
  --deploy_ready

echo "✅ Training workflow completed!"
```

### Run Complete Workflow

```bash
chmod +x complete_training_workflow.sh
./complete_training_workflow.sh
```

---

## 🚨 Troubleshooting Guide

### Common Issues & Solutions

**GPU Out of Memory**:

```bash
# Reduce batch size
--batch_size 8  # Instead of 16

# Enable gradient checkpointing
--gradient_checkpointing

# Use mixed precision
--mixed_precision
```

**Training Not Converging**:

```bash
# Adjust learning rate
--learning_rate 1e-5  # Reduce if loss explodes
--learning_rate 5e-4  # Increase if too slow

# Change scheduler
--scheduler cosine_annealing
--warmup_steps 1000
```

**Poor Generation Quality**:

```bash
# Increase model capacity
--d_model 512  # From 256
--num_layers 8  # From 6

# Improve data augmentation
--augmentation_factor 20  # More variety
```

**Slow Training**:

```bash
# Enable optimizations
--compile_model  # PyTorch 2.0+ compilation
--mixed_precision
--dataloader_num_workers 8
```

---

## 📈 Expected Results by Phase

### Phase 1 (2K samples, $20-50/month)

- **Training Time**: 2-4 hours
- **Layout Accuracy**: 75-80%
- **Use Case**: Prototyping, proof-of-concept
- **Model Size**: 1.2M parameters

### Phase 2 (8K samples, $100-200/month)

- **Training Time**: 6-12 hours
- **Layout Accuracy**: 82-87%
- **Use Case**: MVP, small production
- **Model Size**: 3.6M parameters

### Phase 3 (50K samples, $300-500/month)

- **Training Time**: 1-3 days
- **Layout Accuracy**: 88-92%
- **Use Case**: Production service
- **Model Size**: 12M parameters

### Phase 4 (200K+ samples, $800-1500/month)

- **Training Time**: 3-7 days
- **Layout Accuracy**: 92-96%
- **Use Case**: Enterprise deployment
- **Model Size**: 50M+ parameters

---

## 🎯 Next Steps

After completing training:

1. **Deploy Model**: Use the inference pipeline from Step 4
2. **API Integration**: Set up RESTful API endpoints
3. **Monitoring**: Implement production monitoring
4. **Continuous Learning**: Set up incremental training pipeline
5. **A/B Testing**: Compare against baseline models

---

This guide provides a complete, production-ready training pipeline that automatically scales based on your dataset size and computational resources. The Step 5 implementation handles all the complexity of phase-specific optimization automatically.
