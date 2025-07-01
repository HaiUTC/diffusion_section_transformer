# 🚀 Diffusion Section Transformer - Training Guide

Complete guide for training DLT-style models with automatic phase detection and cost optimization.

## 🎯 Quick Start (5 minutes)

**For immediate testing with sample data:**

```bash
# Clone and setup (if not done already)
git clone <your-repo>
cd diffusion_section_transformer
pip install -r requirements.txt

# Run quick start demo (creates sample data + trains Phase 1 model)
./scripts/quick_start_example.sh
```

**For your own dataset:**

```bash
# Complete automated workflow
./scripts/complete_training_workflow.sh \
  --dataset_dir data/your_dataset \
  --output_dir experiments \
  --experiment_name my_first_model
```

## 📊 Automatic Phase Detection & Cost Optimization

The system automatically detects the optimal training strategy based on your dataset size:

| Dataset Size | Phase   | Training Time | Cost/Month | GPU Requirements | Use Case         |
| ------------ | ------- | ------------- | ---------- | ---------------- | ---------------- |
| **0-2K**     | Phase 1 | 2-4 hours     | $20-50     | 1x T4 (4GB)      | Prototyping, POC |
| **2.5K-10K** | Phase 2 | 6-12 hours    | $100-200   | 1x V100 (16GB)   | MVP, Small Prod  |
| **25K-100K** | Phase 3 | 1-3 days      | $300-500   | 2-4x A100 (40GB) | Production       |
| **100K+**    | Phase 4 | 3-7 days      | $800-1500  | 4-8x A100 (80GB) | Enterprise       |

### 🎯 Expected Performance by Phase

- **Phase 1**: 75-80% layout accuracy
- **Phase 2**: 82-87% layout accuracy
- **Phase 3**: 88-92% layout accuracy
- **Phase 4**: 92-96% layout accuracy

## 📁 Dataset Format

Your dataset should follow this structure:

```
dataset/
├── train/
│   ├── example_0001/
│   │   ├── screenshot.png      # 512x512+ webpage screenshot
│   │   ├── structure.json      # HTML object format
│   │   └── layout.json         # Target section layout
│   └── example_0002/
│       └── ...
├── val/
│   └── ... (same structure)
└── test/
    └── ... (same structure)
```

### Example Files:

**structure.json** (input HTML structure):

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

**layout.json** (target section layout):

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

## 🛠️ Available Scripts

### 1. Complete Automated Workflow

```bash
./scripts/complete_training_workflow.sh \
  --dataset_dir data/your_dataset \
  --output_dir experiments \
  --experiment_name production_model_v1
```

**Options:**

- `--force_phase phase2` - Force specific training phase
- `--skip_validation` - Skip dataset validation
- `--skip_training` - Skip training (evaluation only)
- `--skip_evaluation` - Skip evaluation
- `--config custom_config.yaml` - Use custom configuration

### 2. Manual Training Only

```bash
python3 scripts/train_model.py \
  --dataset_dir data/your_dataset \
  --output_dir models/experiment_1 \
  --auto_phase \
  --epochs 100 \
  --batch_size 16
```

### 3. Model Evaluation Only

```bash
python3 scripts/evaluate_model.py \
  --model_path models/best_model.pth \
  --test_dir data/test \
  --output_dir results/evaluation \
  --evaluate_visual_similarity \
  --evaluate_aesthetics
```

### 4. Quick Start Demo

```bash
./scripts/quick_start_example.sh
```

## 🔧 Phase-Specific Features

### Phase 1: Micro-Scale (0-2K samples)

- **50x aggressive data augmentation**
- **Few-shot learning** with transfer learning
- **Variance-aware loss scheduling**
- **Cost**: $20-50/month (Google Colab Pro sufficient)

### Phase 2: Small-Scale (2.5K-10K samples)

- **3-stage curriculum learning**
- **Two-stage divide-and-conquer training**
- **Progressive data dropout** (20% cost reduction)
- **Cost**: $100-200/month (AWS p3.2xlarge)

### Phase 3: Medium-Scale (25K-100K samples)

- **Standard diffusion training** with CFG=7.5
- **Mixed-precision training** (FP16)
- **Multi-GPU support**
- **Cost**: $300-500/month (AWS p3.8xlarge)

### Phase 4: Large-Scale (100K+ samples)

- **Distributed training** on 4-8 GPUs
- **Gradient accumulation** for large batches
- **Exponential Moving Average** (EMA) of weights
- **Cost**: $800-1500/month (Multi-A100 cluster)

## 📊 Monitoring & Results

### Real-time Monitoring

```bash
# TensorBoard
tensorboard --logdir experiments/your_experiment/models/logs

# Training progress
python3 scripts/monitor_training.py --experiment_dir experiments/your_experiment/models
```

### Evaluation Reports

After training, you'll get:

- **HTML Report**: `results/evaluation/evaluation_report.html`
- **Production Readiness**: `results/production_readiness.md`
- **Raw Metrics**: `results/evaluation/evaluation_results.json`

### Production Readiness Checklist

- ✅ **Layout Accuracy > 85%**
- ✅ **Inference Time < 0.5s**
- ✅ **Model Size < 100M parameters**
- ✅ **GPU Memory < 8GB for inference**

## 🚀 Advanced Usage

### Distributed Training (Phase 3-4)

```bash
# Multi-GPU training
python3 -m torch.distributed.launch \
  --nproc_per_node=4 \
  scripts/train_model.py \
  --dataset_dir data/large_dataset \
  --output_dir models/distributed_training \
  --phase phase4 \
  --distributed
```

### Hyperparameter Optimization

```bash
python3 scripts/hyperparameter_optimization.py \
  --dataset_dir data/your_dataset \
  --output_dir models/hpo_experiment \
  --phase phase2 \
  --num_trials 50 \
  --optimization_method bayesian
```

### Custom Configuration

Create `custom_config.yaml`:

```yaml
training:
  epochs: 150
  batch_size: 32
  learning_rate: 1e-4
  scheduler: cosine_annealing

model:
  d_model: 512
  num_layers: 8
  num_heads: 8

augmentation:
  factor: 30
  rotation_range: 15
  color_jitter: 0.3
```

## 🚨 Troubleshooting

### GPU Out of Memory

```bash
# Reduce batch size
--batch_size 8

# Enable gradient checkpointing
--gradient_checkpointing

# Use mixed precision
--mixed_precision
```

### Training Not Converging

```bash
# Adjust learning rate
--learning_rate 1e-5  # Lower if loss explodes
--learning_rate 5e-4  # Higher if too slow

# Change scheduler
--scheduler cosine_annealing --warmup_steps 1000
```

### Poor Generation Quality

```bash
# Increase model capacity
--d_model 512 --num_layers 8

# Improve data augmentation
--augmentation_factor 20
```

## 📈 Expected Workflow Timeline

### Phase 1 (2K samples)

```
Dataset Prep: 30 minutes
Training: 2-4 hours
Evaluation: 15 minutes
Total: ~5 hours
```

### Phase 2 (8K samples)

```
Dataset Prep: 1 hour
Training: 6-12 hours
Evaluation: 30 minutes
Total: ~8-14 hours
```

### Phase 3 (50K samples)

```
Dataset Prep: 2-4 hours
Training: 1-3 days
Evaluation: 1-2 hours
Total: ~2-4 days
```

### Phase 4 (200K+ samples)

```
Dataset Prep: 4-8 hours
Training: 3-7 days
Evaluation: 2-4 hours
Total: ~4-8 days
```

## 🎯 Production Deployment

After training, deploy using Step 4 inference pipeline:

```bash
# Test inference
python3 examples/step4_inference_demo.py \
  --model_path experiments/your_experiment/models/best_model.pth \
  --input_dir test_inputs/

# Production API server
python3 src/api/server.py \
  --model_path experiments/your_experiment/models/best_model.pth \
  --port 8000
```

## 📚 Additional Resources

- **Detailed Training Guide**: `docs/training_guide.md`
- **Step 5 Implementation Details**: `docs/step5_completion_summary.md`
- **API Documentation**: `docs/api_documentation.md`
- **Performance Benchmarks**: `docs/performance_benchmarks.md`

## 💡 Tips for Success

1. **Start Small**: Begin with Phase 1 to validate your pipeline
2. **Monitor Costs**: Use AWS Spot instances for 60-70% cost savings
3. **Data Quality**: Clean, consistent data > large dataset size
4. **Early Stopping**: Set patience=20 to avoid overtraining
5. **A/B Testing**: Compare against baseline models
6. **Incremental Learning**: Start with general model, fine-tune on your domain

---

## 🆘 Support

If you encounter issues:

1. Check the logs in `experiments/your_experiment/logs/workflow.log`
2. Review the troubleshooting guide above
3. Run the quick start demo to verify setup
4. Check GPU memory and disk space

**Ready to train your first model?** Run `./scripts/quick_start_example.sh` to get started! 🚀
