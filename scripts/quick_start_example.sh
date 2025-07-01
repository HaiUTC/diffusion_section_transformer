#!/bin/bash
# quick_start_example.sh
# Quick Start Example for Diffusion Section Transformer Training
#
# This script demonstrates how to run the complete training workflow
# with sample/demo data to verify everything is working correctly.

set -e

echo "🚀 Quick Start Example - Diffusion Section Transformer"
echo "======================================================="

# Check if we're in the right directory
if [[ ! -f "scripts/train_model.py" ]]; then
    echo "❌ Please run this script from the project root directory"
    echo "   cd /path/to/diffusion_section_transformer"
    echo "   ./scripts/quick_start_example.sh"
    exit 1
fi

# Create sample dataset if it doesn't exist
SAMPLE_DATA_DIR="data/sample_dataset"
if [[ ! -d "$SAMPLE_DATA_DIR" ]]; then
    echo "📁 Creating sample dataset..."
    
    # Create the directory structure
    mkdir -p "$SAMPLE_DATA_DIR"/{train,val,test}
    
    # Create sample training data (50 examples for Phase 1)
    for split in train val test; do
        case $split in
            train) num_examples=35 ;;
            val) num_examples=8 ;;
            test) num_examples=7 ;;
        esac
        
        for i in $(seq 1 $num_examples); do
            example_dir="$SAMPLE_DATA_DIR/$split/example_$(printf "%04d" $i)"
            mkdir -p "$example_dir"
            
            # Create sample screenshot (placeholder)
            convert -size 512x512 xc:white -fill black -gravity center \
                -pointsize 24 -annotate +0+0 "Sample Layout $i" \
                "$example_dir/screenshot.png" 2>/dev/null || \
            python3 -c "
from PIL import Image, ImageDraw, ImageFont
import os
img = Image.new('RGB', (512, 512), 'white')
draw = ImageDraw.Draw(img)
draw.text((200, 250), f'Sample Layout $i', fill='black')
img.save('$example_dir/screenshot.png')
"
            
            # Create sample structure.json
            cat > "$example_dir/structure.json" << EOF
{
  "div.container": {
    "h1.heading": {"text": "Sample Heading $i"},
    "p.paragraph": {"text": "Sample paragraph text for example $i"},
    "div.grid": {
      "div.column": {"text": "Column 1"},
      "div.column": {"text": "Column 2"}
    }
  }
}
EOF
            
            # Create sample layout.json
            cat > "$example_dir/layout.json" << EOF
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
  "props": {}
}
EOF
        done
    done
    
    echo "✅ Sample dataset created with 50 examples"
else
    echo "📁 Using existing sample dataset"
fi

# Run the complete training workflow
echo ""
echo "🏋️ Running Complete Training Workflow"
echo "--------------------------------------"

EXPERIMENT_NAME="quick_start_$(date +%H%M%S)"
OUTPUT_DIR="experiments"

echo "📊 Experiment: $EXPERIMENT_NAME"
echo "📁 Dataset: $SAMPLE_DATA_DIR (50 samples → Phase 1)"
echo "💰 Expected cost: $20-50/month"
echo "⏱️ Expected time: 2-4 hours (demo will be much faster)"

# Ask user for confirmation
echo ""
read -p "Continue with training? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "⏭️ Training cancelled by user"
    exit 0
fi

# Run the workflow
echo "🚀 Starting workflow..."
./scripts/complete_training_workflow.sh \
    --dataset_dir "$SAMPLE_DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --experiment_name "$EXPERIMENT_NAME"

# Check if workflow completed successfully
if [[ $? -eq 0 ]]; then
    echo ""
    echo "🎉 Quick start example completed successfully!"
    echo "============================================="
    echo ""
    echo "📁 Results are in: $OUTPUT_DIR/$EXPERIMENT_NAME/"
    echo ""
    echo "📊 View evaluation report:"
    echo "   open $OUTPUT_DIR/$EXPERIMENT_NAME/results/evaluation/evaluation_report.html"
    echo ""
    echo "📋 Check experiment summary:"
    echo "   cat $OUTPUT_DIR/$EXPERIMENT_NAME/summary.md"
    echo ""
    echo "🔄 To run with your own dataset:"
    echo "   ./scripts/complete_training_workflow.sh \\"
    echo "     --dataset_dir /path/to/your/dataset \\"
    echo "     --output_dir experiments \\"
    echo "     --experiment_name my_experiment"
    echo ""
else
    echo "❌ Quick start example failed"
    echo "   Check logs in: $OUTPUT_DIR/$EXPERIMENT_NAME/logs/workflow.log"
    exit 1
fi 