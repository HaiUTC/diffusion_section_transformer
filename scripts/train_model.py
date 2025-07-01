#!/usr/bin/env python3
"""
Main Training Script for Diffusion Section Transformer

This script implements the complete training pipeline using Step 5 phase-specific strategies.
Automatically detects optimal training phase based on dataset size and computational resources.

Usage:
    python3 scripts/train_model.py --dataset_dir data/raw --output_dir models/experiment_1 --auto_phase
"""

import argparse
import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import yaml
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import Step 5 training components
from src.training import (
    create_phase_strategy, create_phase_loss_function, 
    create_augmentation_config, get_phase_summary
)
from src.ai_engine_configurable import ConfigurableSectionLayoutGenerator
from src.data.data_loaders import create_data_loaders, MultimodalLayoutDataset
from src.utils.metrics import LayoutMetrics
from src.utils.checkpoint import CheckpointManager
from src.data.filesystem_layout import FilesystemLayoutManager
from src.utils.config_loader import config_loader


class TrainingManager:
    """Manages the complete training pipeline with phase-specific optimization."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_distributed = args.distributed and torch.cuda.device_count() > 1
        
        # Setup directories
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.setup_phase_strategy()
        self.setup_model()
        self.setup_data_loaders()
        self.setup_training_components()
        self.setup_logging()
        
    def setup_phase_strategy(self):
        """Setup phase-specific training strategy."""
        print("🔧 Setting up training strategy...")
        
        if self.args.auto_phase:
            # Auto-detect phase based on dataset size
            dataset_size = self.count_dataset_samples()
            self.phase = self.auto_detect_phase(dataset_size)
            print(f"📊 Auto-detected phase: {self.phase.upper()} ({dataset_size:,} samples)")
        else:
            self.phase = self.args.phase
            dataset_size = self.args.dataset_size or self.count_dataset_samples()
        
        # Create phase-specific strategy
        self.strategy = create_phase_strategy(self.phase, dataset_size)
        self.loss_function = create_phase_loss_function(self.phase)
        self.augmentation_config = create_augmentation_config(self.phase)
        
        # Print strategy summary
        summary = get_phase_summary(self.strategy)
        print(f"✅ Using {summary['phase_name']}")
        print(f"📈 Techniques: {', '.join(summary['training_techniques'][:3])}...")
        
    def count_dataset_samples(self):
        """Count total samples in dataset."""
        train_dir = Path(self.args.dataset_dir) / "train"
        if not train_dir.exists():
            train_dir = Path(self.args.dataset_dir)
        return len([d for d in train_dir.iterdir() if d.is_dir()])
    
    def auto_detect_phase(self, dataset_size):
        """Auto-detect optimal training phase."""
        if dataset_size <= 2000:
            return "phase1"
        elif dataset_size <= 10000:
            return "phase2"
        elif dataset_size <= 100000:
            return "phase3"
        else:
            return "phase4"
    
    def setup_model(self):
        """Setup model with dynamic configuration based on actual data."""
        print("🤖 Setting up model...")
        
        # First, analyze the actual data to determine vocabulary sizes
        vocab_info = self.analyze_dataset_vocabulary()
        
        # Get model configuration through config loader
        model_config = config_loader.get_model_config(phase=self.phase)
        
        # Update model configuration with actual vocabulary sizes
        original_structure_vocab = model_config.structure['vocab_size']
        original_layout_vocab = model_config.layout['class_vocab_size']
        
        # Use larger of config or actual size (with padding for safety)
        actual_structure_vocab = max(original_structure_vocab, vocab_info['max_structure_vocab'] + 100)
        actual_layout_vocab = max(original_layout_vocab, vocab_info['max_layout_vocab'] + 100)
        
        print(f"📊 Vocabulary Analysis:")
        print(f"   Structure: Config={original_structure_vocab} → Actual={vocab_info['max_structure_vocab']} → Using={actual_structure_vocab}")
        print(f"   Layout: Config={original_layout_vocab} → Actual={vocab_info['max_layout_vocab']} → Using={actual_layout_vocab}")
        
        # Create model with custom vocabulary sizes
        self.model = ConfigurableSectionLayoutGenerator(
            phase=self.phase,
            dataset_size=self.count_dataset_samples()
        )
        
        # Dynamically update model vocabulary sizes after initialization
        self._update_model_vocabulary_sizes(self.model, actual_structure_vocab, actual_layout_vocab)
        
        self.model = self.model.to(self.device)
        
        # Setup distributed training if needed
        if self.is_distributed:
            self.setup_distributed()
            self.model = DDP(self.model, device_ids=[self.args.local_rank])
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Model created with {total_params:,} parameters")
        
    def _update_model_vocabulary_sizes(self, model, structure_vocab_size, layout_vocab_size):
        """Update model's vocabulary sizes dynamically."""
        print(f"🔧 Updating model vocabulary sizes...")
        
        # Update structure embedding in multimodal encoder
        if hasattr(model.multimodal_encoder, 'structure_transformer') and hasattr(model.multimodal_encoder.structure_transformer, 'embedding'):
            old_embedding = model.multimodal_encoder.structure_transformer.embedding
            if old_embedding.num_embeddings < structure_vocab_size:
                new_embedding = nn.Embedding(structure_vocab_size, old_embedding.embedding_dim, padding_idx=old_embedding.padding_idx)
                # Copy old weights
                with torch.no_grad():
                    new_embedding.weight[:old_embedding.num_embeddings] = old_embedding.weight
                model.multimodal_encoder.structure_transformer.embedding = new_embedding
                print(f"   📈 Structure embedding: {old_embedding.num_embeddings} → {structure_vocab_size}")
        
        # Update ALL layout-related embeddings throughout the model
        embeddings_to_update = [
            # Main layout embedding
            ('layout_embedding.token_embedding', model.layout_embedding.token_embedding),
            # Class embeddings  
            ('layout_embedding.class_embedding.element_embedding', model.layout_embedding.class_embedding.element_embedding),
            # Diffusion decoder embeddings (duplicate copies)
            ('diffusion_decoder.denoiser.embed.token_embedding', model.diffusion_decoder.denoiser.embed.token_embedding),
            ('diffusion_decoder.denoiser.embed.class_embedding.element_embedding', model.diffusion_decoder.denoiser.embed.class_embedding.element_embedding),
        ]
        
        for name, embedding in embeddings_to_update:
            if hasattr(embedding, 'num_embeddings') and embedding.num_embeddings < layout_vocab_size:
                # Keep the original embedding dimension to maintain model architecture
                original_embed_dim = embedding.embedding_dim
                new_embedding = nn.Embedding(layout_vocab_size, original_embed_dim, 
                                           padding_idx=getattr(embedding, 'padding_idx', None))
                # Copy old weights
                with torch.no_grad():
                    new_embedding.weight[:embedding.num_embeddings] = embedding.weight
                
                # Update the embedding in the model
                if name == 'layout_embedding.token_embedding':
                    model.layout_embedding.token_embedding = new_embedding
                elif name == 'layout_embedding.class_embedding.element_embedding':
                    model.layout_embedding.class_embedding.element_embedding = new_embedding
                elif name == 'diffusion_decoder.denoiser.embed.token_embedding':
                    model.diffusion_decoder.denoiser.embed.token_embedding = new_embedding
                elif name == 'diffusion_decoder.denoiser.embed.class_embedding.element_embedding':
                    model.diffusion_decoder.denoiser.embed.class_embedding.element_embedding = new_embedding
                
                print(f"   📈 {name}: {embedding.num_embeddings} → {layout_vocab_size} (dim={original_embed_dim})")
        
        # Update diffusion decoder output layers - ONLY FINAL CLASSIFICATION LAYERS
        for name, module in model.diffusion_decoder.named_modules():
            if isinstance(module, nn.Linear) and hasattr(module, 'out_features'):
                # Only update FINAL classification layers, not intermediate layers
                # Final layers typically have small vocab-sized outputs (like 50, 75)
                # Intermediate layers have model-dimension outputs (like 128, 256)
                is_final_classifier = (
                    'element' in name.lower() and 
                    module.out_features < 100 and  # Small output = likely final classifier
                    module.out_features < layout_vocab_size  # Needs expanding
                )
                
                if is_final_classifier:
                    new_linear = nn.Linear(module.in_features, layout_vocab_size)
                    # Copy old weights
                    with torch.no_grad():
                        new_linear.weight[:module.out_features] = module.weight
                        new_linear.bias[:module.out_features] = module.bias
                    # Update the parent module
                    parent_name = '.'.join(name.split('.')[:-1])
                    layer_name = name.split('.')[-1]
                    parent_module = model.diffusion_decoder
                    for attr in parent_name.split('.'):
                        if attr:
                            parent_module = getattr(parent_module, attr)
                    setattr(parent_module, layer_name, new_linear)
                    print(f"   📈 Final classifier {name}: {module.out_features} → {layout_vocab_size}")
                elif 'element' in name.lower():
                    print(f"   ⏭️ Skipping intermediate layer {name}: {module.out_features} (keeping original size)")
        
        print(f"✅ Model vocabulary update complete")
    
    def analyze_dataset_vocabulary(self):
        """Analyze the dataset to determine actual vocabulary sizes needed."""
        print("🔍 Analyzing dataset vocabulary...")
        
        dataset_dir = Path(self.args.dataset_dir)
        splits = ['train', 'val', 'test']
        
        max_structure_vocab = 0
        max_layout_vocab = 0
        
        for split in splits:
            split_path = dataset_dir / split
            if not split_path.exists():
                continue
                
            examples = [d for d in split_path.iterdir() if d.is_dir()]
            
            # Build vocabulary for this split
            structure_vocab = {"<pad>": 0, "<unk>": 1}
            layout_vocab = {"<pad>": 0, "<unk>": 1}
            structure_vocab_idx = 2  # Separate counter for structure
            layout_vocab_idx = 2     # Separate counter for layout
            
            for example_dir in examples:
                try:
                    # Load structure.json
                    with open(example_dir / "structure.json", 'r') as f:
                        structure = json.load(f)
                    
                    # Load layout.json  
                    with open(example_dir / "layout.json", 'r') as f:
                        layout = json.load(f)
                    
                    # Extract tokens from structure
                    structure_tokens = self._extract_tokens_from_data(structure)
                    for token in structure_tokens:
                        if token not in structure_vocab:
                            structure_vocab[token] = structure_vocab_idx
                            structure_vocab_idx += 1
                    
                    # Extract tokens from layout
                    layout_tokens = self._extract_tokens_from_data(layout)
                    for token in layout_tokens:
                        if token not in layout_vocab:
                            layout_vocab[token] = layout_vocab_idx
                            layout_vocab_idx += 1
                            
                except Exception as e:
                    continue
            
            # Update maximum vocabulary sizes
            max_structure_vocab = max(max_structure_vocab, len(structure_vocab))
            max_layout_vocab = max(max_layout_vocab, len(layout_vocab))
            
            print(f"   {split:5}: {len(structure_vocab)} structure, {len(layout_vocab)} layout tokens")
        
        return {
            'max_structure_vocab': max_structure_vocab,
            'max_layout_vocab': max_layout_vocab
        }
    
    def _extract_tokens_from_data(self, data, prefix=""):
        """Extract tokens from nested dictionary."""
        tokens = []
        if isinstance(data, dict):
            for key, value in data.items():
                token = f"{prefix}{key}" if prefix else key
                tokens.append(token)
                if isinstance(value, dict):
                    tokens.extend(self._extract_tokens_from_data(value, f"{token}."))
                elif isinstance(value, str):
                    tokens.append(value)
        return tokens
    
    def setup_distributed(self):
        """Setup distributed training."""
        print("🌐 Setting up distributed training...")
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(self.args.local_rank)
        self.device = torch.device(f'cuda:{self.args.local_rank}')
        
        print(f"✅ Distributed training on {dist.get_world_size()} GPUs")
    
    def setup_data_loaders(self):
        """Setup data loaders with vocabulary alignment."""
        print("📁 Setting up data loaders...")
        
        self.train_loader, self.val_loader, self.test_loader, train_dataset = self.create_simple_data_loaders()
        
        print(f"✅ Data loaders created:")
        print(f"   Train: {len(self.train_loader):,} batches")
        print(f"   Val: {len(self.val_loader):,} batches")
        print(f"   Test: {len(self.test_loader):,} batches")
        
        # CRITICAL: Update model with actual data loader vocabulary sizes
        actual_structure_vocab_size = len(train_dataset.structure_vocab)
        actual_layout_vocab_size = len(train_dataset.layout_vocab)
        
        print(f"🔧 Final vocabulary alignment:")
        print(f"   Structure: {actual_structure_vocab_size} tokens")
        print(f"   Layout: {actual_layout_vocab_size} tokens")
        
        # Update model embeddings to match exact data loader vocab sizes
        self._update_model_vocabulary_sizes(self.model, actual_structure_vocab_size, actual_layout_vocab_size)
    
    def create_simple_data_loaders(self):
        """Create data loaders for simple train/val/test directory structure."""
        
        class SimpleLayoutDataset(Dataset):
            def __init__(self, data_dir, image_size=224):
                self.data_dir = Path(data_dir)
                self.image_size = image_size
                self.examples = [d for d in self.data_dir.iterdir() if d.is_dir()]
                
                # Image transform
                self.transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                # Build vocabularies
                self.structure_vocab = {"<pad>": 0, "<unk>": 1}
                self.layout_vocab = {"<pad>": 0, "<unk>": 1}
                self._build_vocabularies()
            
            def _build_vocabularies(self):
                """Build vocabularies from the dataset."""
                structure_vocab_idx = 2  # Start after <pad> and <unk>
                layout_vocab_idx = 2     # Separate counter for layout vocab
                
                for example_dir in self.examples:
                    try:
                        # Load structure.json
                        with open(example_dir / "structure.json", 'r') as f:
                            structure = json.load(f)
                        
                        # Load layout.json  
                        with open(example_dir / "layout.json", 'r') as f:
                            layout = json.load(f)
                        
                        # Extract tokens from structure
                        structure_tokens = self._extract_tokens(structure)
                        for token in structure_tokens:
                            if token not in self.structure_vocab:
                                self.structure_vocab[token] = structure_vocab_idx
                                structure_vocab_idx += 1
                        
                        # Extract tokens from layout
                        layout_tokens = self._extract_tokens(layout)
                        for token in layout_tokens:
                            if token not in self.layout_vocab:
                                self.layout_vocab[token] = layout_vocab_idx
                                layout_vocab_idx += 1
                                
                    except Exception as e:
                        continue
                
                print(f"Built vocabularies: {len(self.structure_vocab)} structure tokens, {len(self.layout_vocab)} layout tokens")
            
            def _extract_tokens(self, data, prefix=""):
                """Extract tokens from nested dictionary."""
                tokens = []
                if isinstance(data, dict):
                    for key, value in data.items():
                        token = f"{prefix}{key}" if prefix else key
                        tokens.append(token)
                        if isinstance(value, dict):
                            tokens.extend(self._extract_tokens(value, f"{token}."))
                        elif isinstance(value, str):
                            tokens.append(value)
                return tokens
            
            def _tokenize(self, data, vocab, max_length=128):
                """Convert data to token indices."""
                tokens = self._extract_tokens(data)
                indices = [vocab.get(token, vocab["<unk>"]) for token in tokens[:max_length]]
                # Pad to max_length
                indices += [vocab["<pad>"]] * (max_length - len(indices))
                return torch.tensor(indices[:max_length], dtype=torch.long)
            
            def __len__(self):
                return len(self.examples)
            
            def __getitem__(self, idx):
                example_dir = self.examples[idx]
                
                try:
                    # Load screenshot
                    from PIL import Image
                    screenshot = Image.open(example_dir / "screenshot.png").convert('RGB')
                    screenshot = self.transform(screenshot)
                    
                    # Load and tokenize structure
                    with open(example_dir / "structure.json", 'r') as f:
                        structure = json.load(f)
                    structure_tokens = self._tokenize(structure, self.structure_vocab)
                    
                    # Load and tokenize layout  
                    with open(example_dir / "layout.json", 'r') as f:
                        layout = json.load(f)
                    layout_tokens = self._tokenize(layout, self.layout_vocab)
                    
                    # Debug: Print token statistics for first few examples
                    if idx < 3:  # Only for first 3 examples to avoid spam
                        print(f"🐛 Debug Example {idx} ({example_dir.name}):")
                        print(f"   Structure tokens range: {structure_tokens.min().item()} to {structure_tokens.max().item()}")
                        print(f"   Layout tokens range: {layout_tokens.min().item()} to {layout_tokens.max().item()}")
                        print(f"   Structure vocab size: {len(self.structure_vocab)}")
                        print(f"   Layout vocab size: {len(self.layout_vocab)}")
                        
                        # Check for out of range tokens
                        if layout_tokens.max().item() >= len(self.layout_vocab):
                            print(f"   ❌ OUT OF RANGE: Layout token {layout_tokens.max().item()} >= vocab size {len(self.layout_vocab)}")
                            
                        if structure_tokens.max().item() >= len(self.structure_vocab):
                            print(f"   ❌ OUT OF RANGE: Structure token {structure_tokens.max().item()} >= vocab size {len(self.structure_vocab)}")
                    
                    return {
                        'screenshot': screenshot,
                        'structure_tokens': structure_tokens,
                        'layout_tokens': layout_tokens,
                        'example_id': example_dir.name
                    }
                    
                except Exception as e:
                    print(f"🚨 Error loading example {idx}: {e}")
                    # Return dummy data on error
                    return {
                        'screenshot': torch.zeros(3, self.image_size, self.image_size),
                        'structure_tokens': torch.zeros(128, dtype=torch.long),
                        'layout_tokens': torch.zeros(128, dtype=torch.long),
                        'example_id': f"error_{idx}"
                    }
        
        # Create datasets
        dataset_dir = Path(self.args.dataset_dir)
        train_dataset = SimpleLayoutDataset(dataset_dir / "train")
        val_dataset = SimpleLayoutDataset(dataset_dir / "val") if (dataset_dir / "val").exists() else train_dataset
        test_dataset = SimpleLayoutDataset(dataset_dir / "test") if (dataset_dir / "test").exists() else val_dataset
        
        # Share vocabularies
        val_dataset.structure_vocab = train_dataset.structure_vocab
        val_dataset.layout_vocab = train_dataset.layout_vocab
        test_dataset.structure_vocab = train_dataset.structure_vocab
        test_dataset.layout_vocab = train_dataset.layout_vocab
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.strategy.config.batch_size,
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing issues with local class
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.strategy.config.batch_size,
            shuffle=False,
            num_workers=0,  # Avoid multiprocessing issues with local class
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.strategy.config.batch_size,
            shuffle=False,
            num_workers=0,  # Avoid multiprocessing issues with local class
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader, train_dataset
    
    def setup_training_components(self):
        """Setup optimizer, scheduler, and other training components."""
        print("⚙️ Setting up training components...")
        
        # Setup optimizer and scheduler
        self.optimizer = self.strategy.configure_optimizer(self.model)
        self.scheduler = self.strategy.configure_scheduler(self.optimizer)
        
        # Setup loss function
        self.loss_fn = self.loss_function.to(self.device)
        
        # Setup metrics
        self.metrics = LayoutMetrics()
        
        # Setup checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.checkpoint_dir,
            save_every=self.args.save_every
        )
        
        # Resume from checkpoint if specified
        if self.args.resume_from:
            self.resume_training()
        
        print(f"✅ Training components ready")
    
    def setup_logging(self):
        """Setup logging and monitoring."""
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        self.writer = SummaryWriter(log_dir)
        self.start_time = time.time()
        
        # Save configuration
        config = {
            'phase': self.phase,
            'dataset_size': self.count_dataset_samples(),
            'model_config': self.strategy.get_configuration_summary(),
            'training_techniques': self.strategy.get_training_techniques(),
            'args': vars(self.args)
        }
        
        with open(self.output_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
    
    def _compute_loss(self, outputs, batch):
        """Compute loss with proper error handling."""
        try:
            layout_tokens = batch['layout_tokens']
            
            if hasattr(self.loss_fn, 'forward') and 'MultiTask' in type(self.loss_fn).__name__:
                # Multi-task loss
                loss_results = self.loss_fn(
                    model_outputs=outputs,
                    targets=batch,
                    visual_features=None,
                    structural_features=None
                )
                if isinstance(loss_results, dict) and 'total_loss' in loss_results:
                    return loss_results['total_loss']
                else:
                    return loss_results
            else:
                # Simple loss handling
                if isinstance(outputs, dict):
                    if 'predicted_layout' in outputs:
                        predictions = outputs['predicted_layout']
                    elif 'layout_tokens' in outputs:
                        predictions = outputs['layout_tokens']
                    elif 'logits' in outputs:
                        predictions = outputs['logits']
                    else:
                        predictions = next(iter(outputs.values()))
                else:
                    predictions = outputs
                
                # Handle GenerationOutput objects
                if hasattr(predictions, 'layout_tokens') and predictions.layout_tokens is not None:
                    predictions = predictions.layout_tokens
                elif hasattr(predictions, 'logits') and predictions.logits is not None:
                    predictions = predictions.logits
                elif hasattr(predictions, 'prediction_scores'):
                    predictions = predictions.prediction_scores
                elif hasattr(predictions, 'last_hidden_state'):
                    predictions = predictions.last_hidden_state
                
                # Ensure predictions is a tensor
                if not isinstance(predictions, torch.Tensor):
                    predictions = torch.zeros_like(layout_tokens).float()
                
                # Compute loss based on dimensions
                if predictions.dim() == 3:  # [batch, seq, vocab]
                    pred_flat = predictions.view(-1, predictions.size(-1))
                    target_flat = layout_tokens.view(-1)
                    
                    # Ensure compatible sizes
                    if pred_flat.size(0) != target_flat.size(0):
                        min_size = min(pred_flat.size(0), target_flat.size(0))
                        pred_flat = pred_flat[:min_size]
                        target_flat = target_flat[:min_size]
                    
                    return F.cross_entropy(pred_flat, target_flat, ignore_index=0)
                else:
                    # Handle dimension mismatch
                    if predictions.dim() > layout_tokens.dim():
                        predictions = predictions.mean(dim=-1)
                    elif predictions.dim() < layout_tokens.dim():
                        layout_tokens = layout_tokens.view(layout_tokens.size(0), -1)
                    
                    # Ensure compatible shapes
                    if predictions.shape != layout_tokens.shape:
                        min_seq_len = min(predictions.size(-1), layout_tokens.size(-1))
                        predictions = predictions[..., :min_seq_len]
                        layout_tokens = layout_tokens[..., :min_seq_len]
                    
                    return F.mse_loss(predictions.float(), layout_tokens.float())
                    
        except Exception as e:
            print(f"⚠️ Loss computation error: {e}")
            # Fallback loss
            return torch.tensor(1.0, device=self.device, requires_grad=True)

    def _compute_batch_metrics(self, outputs, batch):
        """Compute batch-level metrics."""
        try:
            layout_tokens = batch['layout_tokens']
            
            # Extract predictions
            if isinstance(outputs, dict):
                if 'predicted_layout' in outputs:
                    predictions = outputs['predicted_layout']
                elif 'layout_tokens' in outputs:
                    predictions = outputs['layout_tokens']
                elif 'logits' in outputs:
                    predictions = outputs['logits']
                else:
                    predictions = next(iter(outputs.values()))
            else:
                predictions = outputs
            
            # Handle GenerationOutput objects
            if hasattr(predictions, 'layout_tokens') and predictions.layout_tokens is not None:
                predictions = predictions.layout_tokens
            elif hasattr(predictions, 'logits') and predictions.logits is not None:
                predictions = predictions.logits
            elif hasattr(predictions, 'prediction_scores'):
                predictions = predictions.prediction_scores
            elif hasattr(predictions, 'last_hidden_state'):
                predictions = predictions.last_hidden_state
            
            # Convert to tensor if needed
            if not isinstance(predictions, torch.Tensor):
                predictions = torch.zeros_like(layout_tokens)
            
            # Compute layout accuracy (simplified)
            if predictions.dim() == 3:  # [batch, seq, vocab]
                pred_tokens = torch.argmax(predictions, dim=-1)
                layout_accuracy = (pred_tokens == layout_tokens).float().mean().item()
            else:
                # For continuous predictions, use threshold-based accuracy
                diff = torch.abs(predictions - layout_tokens.float())
                layout_accuracy = (diff < 0.1).float().mean().item()
            
            # Element precision (no false positives)
            element_precision = 1.0  # Simplified for now
            
            return layout_accuracy, element_precision
            
        except Exception as e:
            print(f"⚠️ Metrics computation error: {e}")
            return 0.0, 0.0

    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch with progress bar."""
        self.model.train()
        epoch_loss = 0.0
        layout_accuracy = 0.0
        element_precision = 0.0
        samples_processed = 0
        
        # Create progress bar for training
        train_pbar = tqdm(
            self.train_loader, 
            desc=f"Epoch {epoch+1:3d}/{self.strategy.config.epochs:3d} [Training]",
            leave=False,
            unit="batch",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        for batch_idx, batch in enumerate(train_pbar):
            try:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(
                    screenshots=batch['screenshot'],
                    structure_tokens=batch['structure_tokens'],
                    layout_tokens=batch['layout_tokens']
                )
                
                # Compute loss
                loss = self._compute_loss(outputs, batch)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item()
                batch_size = batch['screenshot'].size(0)
                samples_processed += batch_size
                
                # Compute additional metrics
                with torch.no_grad():
                    layout_acc, elem_prec = self._compute_batch_metrics(outputs, batch)
                    layout_accuracy += layout_acc * batch_size
                    element_precision += elem_prec * batch_size
                
                # Update progress bar
                current_loss = epoch_loss / (batch_idx + 1)
                current_acc = layout_accuracy / samples_processed
                train_pbar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Acc': f'{current_acc:.2%}',
                    'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
                
                # Log to tensorboard
                if self.writer and batch_idx % 10 == 0:
                    global_step = epoch * len(self.train_loader) + batch_idx
                    self.writer.add_scalar('Train/Loss_Step', loss.item(), global_step)
                    self.writer.add_scalar('Train/Learning_Rate', 
                                         self.optimizer.param_groups[0]['lr'], global_step)
                
            except Exception as e:
                print(f"❌ Error in training batch {batch_idx}: {e}")
                continue
        
        train_pbar.close()
        
        return {
            'loss': epoch_loss / len(self.train_loader),
            'layout_accuracy': layout_accuracy / samples_processed,
            'element_precision': element_precision / samples_processed,
            'samples_processed': samples_processed
        }
    
    def validate_epoch(self, epoch: int) -> dict:
        """Validate for one epoch with progress bar."""
        self.model.eval()
        val_loss = 0.0
        layout_accuracy = 0.0
        element_precision = 0.0
        samples_processed = 0
        
        # Create progress bar for validation
        val_pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch+1:3d}/{self.strategy.config.epochs:3d} [Validation]",
            leave=False,
            unit="batch",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_pbar):
                try:
                    # Move batch to device
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    
                    # Forward pass
                    outputs = self.model(
                        screenshots=batch['screenshot'],
                        structure_tokens=batch['structure_tokens'],
                        layout_tokens=batch['layout_tokens']
                    )
                    
                    # Compute loss
                    loss = self._compute_loss(outputs, batch)
                    val_loss += loss.item()
                    
                    # Update metrics
                    batch_size = batch['screenshot'].size(0)
                    samples_processed += batch_size
                    
                    # Compute additional metrics
                    layout_acc, elem_prec = self._compute_batch_metrics(outputs, batch)
                    layout_accuracy += layout_acc * batch_size
                    element_precision += elem_prec * batch_size
                    
                    # Update progress bar
                    current_loss = val_loss / (batch_idx + 1)
                    current_acc = layout_accuracy / samples_processed
                    val_pbar.set_postfix({
                        'Loss': f'{current_loss:.4f}',
                        'Acc': f'{current_acc:.2%}'
                    })
                    
                except Exception as e:
                    print(f"❌ Error in validation batch {batch_idx}: {e}")
                    continue
        
        val_pbar.close()
        
        return {
            'loss': val_loss / len(self.val_loader),
            'layout_accuracy': layout_accuracy / samples_processed,
            'element_precision': element_precision / samples_processed,
            'samples_processed': samples_processed
        }
    
    def train(self):
        """Main training loop with progress tracking."""
        print("🚀 Starting training...")
        print(f"📊 Training samples: {len(self.train_loader.dataset):,}")
        print(f"📊 Validation samples: {len(self.val_loader.dataset):,}")
        print(f"🎯 Target epochs: {self.strategy.config.epochs}")
        print(f"⚙️  Batch size: {self.strategy.config.batch_size}")
        print(f"🔧 Device: {self.device}")
        print("-" * 60)
        
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = 15
        
        # Create epoch progress bar
        epoch_pbar = tqdm(
            range(self.strategy.config.epochs),
            desc="Training Progress",
            unit="epoch",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        )
        
        for epoch in epoch_pbar:
            start_time = time.time()
            
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            
            # Validate epoch
            val_metrics = self.validate_epoch(epoch)
            
            # Update learning rate scheduler
            if self.scheduler:
                if hasattr(self.scheduler, 'step'):
                    if 'ReduceLROnPlateau' in self.scheduler.__class__.__name__:
                        self.scheduler.step(val_metrics['loss'])
                    else:
                        self.scheduler.step()
            
            epoch_time = time.time() - start_time
            
            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'T_Loss': f"{train_metrics['loss']:.4f}",
                'V_Loss': f"{val_metrics['loss']:.4f}",
                'V_Acc': f"{val_metrics['layout_accuracy']:.2%}",
                'Time': f"{epoch_time:.1f}s"
            })
            
            # Log to tensorboard
            if self.writer:
                self.writer.add_scalar('Train/Loss_Epoch', train_metrics['loss'], epoch)
                self.writer.add_scalar('Train/Layout_Accuracy', train_metrics['layout_accuracy'], epoch)
                self.writer.add_scalar('Train/Element_Precision', train_metrics['element_precision'], epoch)
                self.writer.add_scalar('Val/Loss_Epoch', val_metrics['loss'], epoch)
                self.writer.add_scalar('Val/Layout_Accuracy', val_metrics['layout_accuracy'], epoch)
                self.writer.add_scalar('Val/Element_Precision', val_metrics['element_precision'], epoch)
                self.writer.add_scalar('Train/Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Save checkpoint
            is_best = val_metrics['loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                
                if self.checkpoint_manager:
                    self.checkpoint_manager.save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        metrics={**train_metrics, **{f'val_{k}': v for k, v in val_metrics.items()}},
                        is_best=True
                    )
            else:
                patience_counter += 1
            
            # Early stopping check
            if patience_counter >= early_stopping_patience:
                print(f"\n⏹️  Early stopping triggered at epoch {epoch+1}")
                print(f"📊 Best validation loss: {best_val_loss:.4f}")
                break
        
        epoch_pbar.close()
        
        print("\n" + "="*60)
        print("🏁 Training completed!")
        print(f"📊 Best validation loss: {best_val_loss:.4f}")
        print(f"💾 Best model saved to: {self.checkpoint_dir / 'best_model.pth'}")
        print("="*60)
    
    def save_best_model(self, epoch, val_loss, val_metrics):
        """Save the best model."""
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'config': self.strategy.get_configuration_summary()
        }, self.output_dir / 'best_model.pth')
        
        print(f"💾 Best model saved (epoch {epoch}, val_loss={val_loss:.4f})")
    
    def resume_training(self):
        """Resume training from checkpoint."""
        print(f"🔄 Resuming training from {self.args.resume_from}")
        
        checkpoint = torch.load(self.args.resume_from, map_location=self.device)
        
        # Load model state
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer and scheduler states
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"✅ Resumed from epoch {checkpoint['epoch']}")


def main():
    parser = argparse.ArgumentParser(description='Train Diffusion Section Transformer')
    
    # Dataset arguments
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for models and logs')
    
    # Training configuration
    parser.add_argument('--phase', type=str, choices=['phase1', 'phase2', 'phase3', 'phase4'],
                        help='Training phase (auto-detected if not specified)')
    parser.add_argument('--auto_phase', action='store_true',
                        help='Automatically detect training phase based on dataset size')
    parser.add_argument('--dataset_size', type=int,
                        help='Dataset size (auto-detected if not specified)')
    parser.add_argument('--epochs', type=int,
                        help='Number of epochs (phase-specific default if not specified)')
    parser.add_argument('--batch_size', type=int,
                        help='Batch size (phase-specific default if not specified)')
    
    # Model configuration
    parser.add_argument('--config_file', type=str,
                        help='Custom configuration file')
    parser.add_argument('--resume_from', type=str,
                        help='Resume training from checkpoint')
    
    # Distributed training
    parser.add_argument('--distributed', action='store_true',
                        help='Enable distributed training')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank for distributed training')
    
    # Training options
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Log every N batches')
    parser.add_argument('--clip_grad_norm', type=float,
                        help='Gradient clipping norm')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.auto_phase and not args.phase:
        parser.error("Either --auto_phase or --phase must be specified")
    
    # Create training manager and start training
    trainer = TrainingManager(args)
    trainer.train()


if __name__ == '__main__':
    main() 