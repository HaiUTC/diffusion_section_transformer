#!/usr/bin/env python3
"""
Inference Demo Script for Diffusion Section Transformer
This script shows how to use a trained model to generate layouts.

Usage:
    python3 scripts/inference_demo.py --model_path experiments/production_50_pairs/models/best_model.pth --input_dir new_examples/
"""

import argparse
import torch
import json
from pathlib import Path
import sys
from PIL import Image
from torchvision import transforms
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ai_engine_configurable import ConfigurableSectionLayoutGenerator


class LayoutInferenceEngine:
    """Easy-to-use inference engine for generating layouts."""
    
    def __init__(self, model_path, device=None):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the trained model (.pth file)
            device: Device to run inference on (auto-detected if None)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        
        # Load model
        self.load_model()
        
        # Setup image preprocessing
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Build vocabulary from training (simplified)
        self.structure_vocab = {"<pad>": 0, "<unk>": 1}
        self.layout_vocab = {"<pad>": 0, "<unk>": 1}
        self._build_basic_vocabulary()
        
        print(f"✅ Inference engine ready on {self.device}")
        
    def load_model(self):
        """Load the trained model."""
        print(f"🔄 Loading model from {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Detect phase from model architecture
        phase = self.detect_phase_from_checkpoint(checkpoint)
        print(f"🎯 Detected model phase: {phase}")
        
        # Create model
        self.model = ConfigurableSectionLayoutGenerator(phase=phase)
        
        # Load state dict (ignore unexpected keys)
        missing_keys, unexpected_keys = self.model.load_state_dict(
            checkpoint['model_state_dict'], strict=False
        )
        
        if unexpected_keys:
            print(f"⚠️ Ignoring {len(unexpected_keys)} unexpected keys")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Store model info
        self.model_info = {
            'phase': phase,
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'epoch': checkpoint.get('epoch', 'unknown'),
            'val_loss': checkpoint.get('val_loss', 'unknown')
        }
        
        print(f"✅ Model loaded: {self.model_info['parameters']:,} parameters")
        
    def detect_phase_from_checkpoint(self, checkpoint):
        """Detect training phase from model architecture."""
        try:
            state_dict = checkpoint['model_state_dict']
            
            # Check vision transformer dimensions
            for key in state_dict.keys():
                if 'vision_transformer.blocks.0.attention.w_q.weight' in key:
                    dim = state_dict[key].shape[0]
                    
                    if dim == 128:
                        return 'phase1'
                    elif dim == 192:
                        return 'phase2'
                    elif dim == 256:
                        return 'phase3'
                    elif dim == 384:
                        return 'phase4'
            
            # Fallback to phase1
            return 'phase1'
            
        except Exception as e:
            print(f"⚠️ Error detecting phase: {e}")
            return 'phase1'
    
    def _build_basic_vocabulary(self):
        """Build a basic vocabulary for inference."""
        # Common HTML/CSS tokens
        common_tokens = [
            'div', 'section', 'header', 'footer', 'nav', 'main', 'article',
            'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'span', 'a', 'img',
            'ul', 'ol', 'li', 'table', 'tr', 'td', 'th', 'form', 'input',
            'button', 'textarea', 'select', 'option', 'label', 'br', 'hr',
            'container', 'wrapper', 'content', 'sidebar', 'menu', 'banner',
            'grid', 'row', 'col', 'column', 'flex', 'block', 'inline',
            'text', 'image', 'link', 'list', 'item', 'card', 'panel'
        ]
        
        vocab_idx = 2  # Start after <pad> and <unk>
        for token in common_tokens:
            if token not in self.structure_vocab:
                self.structure_vocab[token] = vocab_idx
                self.layout_vocab[token] = vocab_idx
                vocab_idx += 1
    
    def preprocess_image(self, image_path):
        """Preprocess screenshot image."""
        image = Image.open(image_path).convert('RGB')
        return self.image_transform(image).unsqueeze(0)  # Add batch dimension
    
    def preprocess_structure(self, structure_data):
        """Convert structure data to tokens."""
        tokens = self._extract_tokens_from_structure(structure_data)
        indices = [self.structure_vocab.get(token, self.structure_vocab["<unk>"]) 
                  for token in tokens[:128]]
        # Pad to 128
        indices += [self.structure_vocab["<pad>"]] * (128 - len(indices))
        return torch.tensor(indices[:128], dtype=torch.long).unsqueeze(0)
    
    def _extract_tokens_from_structure(self, data, prefix=""):
        """Extract tokens from structure data."""
        tokens = []
        if isinstance(data, dict):
            for key, value in data.items():
                token = f"{prefix}{key}" if prefix else key
                tokens.append(token)
                if isinstance(value, dict):
                    tokens.extend(self._extract_tokens_from_structure(value, f"{token}."))
                elif isinstance(value, str) and value.strip():
                    tokens.append(value.strip())
        elif isinstance(data, str):
            tokens.append(data)
        return tokens
    
    def generate_layout(self, screenshot_path, structure_data):
        """
        Generate layout from screenshot and structure.
        
        Args:
            screenshot_path: Path to screenshot image
            structure_data: HTML structure data (dict or JSON string)
            
        Returns:
            Generated layout with confidence scores
        """
        print(f"🔮 Generating layout...")
        start_time = time.time()
        
        # Preprocess inputs
        screenshot = self.preprocess_image(screenshot_path).to(self.device)
        
        if isinstance(structure_data, str):
            structure_data = json.loads(structure_data)
        
        structure_tokens = self.preprocess_structure(structure_data).to(self.device)
        
        # Generate layout
        with torch.no_grad():
            outputs = self.model(
                screenshot=screenshot,
                structure_tokens=structure_tokens,
                training=False
            )
        
        inference_time = time.time() - start_time
        
        # Process outputs
        result = self._process_model_output(outputs, inference_time)
        
        print(f"✅ Layout generated in {inference_time:.3f}s")
        return result
    
    def _process_model_output(self, outputs, inference_time):
        """Process model output into readable format."""
        result = {
            'inference_time': inference_time,
            'model_info': self.model_info,
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if hasattr(outputs, 'elements') and outputs.elements:
            # Structured output
            result['layout_type'] = 'structured'
            result['elements'] = outputs.elements
            result['confidence_scores'] = getattr(outputs, 'confidence_scores', []).tolist() if hasattr(outputs, 'confidence_scores') else []
            result['aesthetic_score'] = getattr(outputs, 'aesthetic_score', 0.0)
            
        elif hasattr(outputs, 'layout_tokens'):
            # Token-based output
            result['layout_type'] = 'tokens'
            result['layout_tokens'] = outputs.layout_tokens.cpu().numpy().tolist()
            result['confidence_scores'] = getattr(outputs, 'confidence_scores', []).tolist() if hasattr(outputs, 'confidence_scores') else []
            
        else:
            # Fallback
            result['layout_type'] = 'raw'
            result['raw_output'] = str(outputs)
        
        return result
    
    def batch_inference(self, input_dir, output_dir):
        """
        Run inference on multiple examples.
        
        Args:
            input_dir: Directory containing input examples
            output_dir: Directory to save results
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all examples
        examples = [d for d in input_path.iterdir() if d.is_dir()]
        
        print(f"🔍 Found {len(examples)} examples to process")
        
        results = []
        for i, example_dir in enumerate(examples, 1):
            try:
                print(f"\n📝 Processing example {i}/{len(examples)}: {example_dir.name}")
                
                # Load screenshot and structure
                screenshot_path = example_dir / "screenshot.png"
                structure_path = example_dir / "structure.json"
                
                if not screenshot_path.exists() or not structure_path.exists():
                    print(f"⚠️ Missing files in {example_dir.name}, skipping...")
                    continue
                
                with open(structure_path, 'r') as f:
                    structure_data = json.load(f)
                
                # Generate layout
                result = self.generate_layout(screenshot_path, structure_data)
                result['example_id'] = example_dir.name
                
                # Save result
                output_file = output_path / f"{example_dir.name}_result.json"
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                results.append(result)
                print(f"✅ Saved result to {output_file}")
                
            except Exception as e:
                print(f"❌ Error processing {example_dir.name}: {e}")
        
        # Save summary
        summary = {
            'total_examples': len(examples),
            'successful_inferences': len(results),
            'model_info': self.model_info,
            'results_summary': {
                'avg_inference_time': sum(r['inference_time'] for r in results) / len(results) if results else 0,
                'total_time': sum(r['inference_time'] for r in results)
            }
        }
        
        with open(output_path / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n🎉 Batch inference completed!")
        print(f"   Processed: {len(results)}/{len(examples)} examples")
        print(f"   Results saved to: {output_path}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Run inference with trained model')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model (.pth file)')
    parser.add_argument('--input_dir', type=str,
                        help='Directory with input examples (for batch inference)')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                        help='Output directory for results')
    parser.add_argument('--screenshot', type=str,
                        help='Single screenshot file (for single inference)')
    parser.add_argument('--structure', type=str,
                        help='Single structure file (for single inference)')
    
    args = parser.parse_args()
    
    # Create inference engine
    engine = LayoutInferenceEngine(args.model_path)
    
    if args.input_dir:
        # Batch inference
        engine.batch_inference(args.input_dir, args.output_dir)
    elif args.screenshot and args.structure:
        # Single inference
        with open(args.structure, 'r') as f:
            structure_data = json.load(f)
        
        result = engine.generate_layout(args.screenshot, structure_data)
        
        # Save result
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / "result.json", 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"✅ Result saved to {output_path / 'result.json'}")
    else:
        parser.error("Either --input_dir or both --screenshot and --structure must be provided")


if __name__ == '__main__':
    main() 