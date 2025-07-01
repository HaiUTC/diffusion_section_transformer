"""
Checkpoint Management for Diffusion Section Transformer
Utilities for saving and loading training checkpoints.
"""

import torch
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime


class CheckpointManager:
    """Manages saving and loading of training checkpoints."""
    
    def __init__(self, checkpoint_dir: Union[str, Path], save_every: int = 10, keep_last: int = 5):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_every: Save checkpoint every N epochs
            keep_last: Keep only the last N checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = save_every
        self.keep_last = keep_last
        
        # Track saved checkpoints
        self.saved_checkpoints = []
    
    def save_checkpoint(self, 
                       epoch: int, 
                       model: torch.nn.Module, 
                       optimizer: torch.optim.Optimizer,
                       scheduler: Optional[Any] = None,
                       loss: float = 0.0,
                       metrics: Optional[Dict[str, Any]] = None,
                       **kwargs) -> Path:
        """
        Save a training checkpoint.
        
        Args:
            epoch: Current epoch number
            model: Model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler (optional)
            loss: Current loss value
            metrics: Evaluation metrics (optional)
            **kwargs: Additional data to save
        
        Returns:
            Path to saved checkpoint
        """
        # Only save if it's time
        if epoch % self.save_every != 0:
            return None
        
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics or {},
            **kwargs
        }
        
        # Add scheduler if provided
        if scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
        
        # Create checkpoint filename
        checkpoint_filename = f"checkpoint_epoch_{epoch:04d}.pth"
        checkpoint_path = self.checkpoint_dir / checkpoint_filename
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        
        # Track saved checkpoint
        self.saved_checkpoints.append({
            'epoch': epoch,
            'path': checkpoint_path,
            'loss': loss,
            'timestamp': datetime.now()
        })
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        
        print(f"💾 Checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        
        Returns:
            Checkpoint data dictionary
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        print(f"📂 Checkpoint loaded: {checkpoint_path}")
        return checkpoint_data
    
    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load the most recent checkpoint.
        
        Returns:
            Latest checkpoint data or None if no checkpoints found
        """
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        if not checkpoints:
            return None
        
        # Sort by epoch number (extracted from filename)
        checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
        latest_checkpoint = checkpoints[-1]
        
        return self.load_checkpoint(latest_checkpoint)
    
    def list_checkpoints(self) -> list:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint information
        """
        checkpoints = []
        for checkpoint_file in self.checkpoint_dir.glob("checkpoint_epoch_*.pth"):
            try:
                # Extract epoch from filename
                epoch = int(checkpoint_file.stem.split('_')[-1])
                checkpoints.append({
                    'epoch': epoch,
                    'path': checkpoint_file,
                    'size': checkpoint_file.stat().st_size,
                    'modified': datetime.fromtimestamp(checkpoint_file.stat().st_mtime)
                })
            except (ValueError, IndexError):
                continue
        
        # Sort by epoch
        checkpoints.sort(key=lambda x: x['epoch'])
        return checkpoints
    
    def save_best_model(self, 
                       model: torch.nn.Module, 
                       epoch: int,
                       metric_value: float,
                       metric_name: str = 'loss',
                       is_better: callable = lambda new, old: new < old) -> bool:
        """
        Save model if it's the best so far.
        
        Args:
            model: Model to potentially save
            epoch: Current epoch
            metric_value: Current metric value
            metric_name: Name of the metric
            is_better: Function to determine if new metric is better
        
        Returns:
            True if model was saved as best
        """
        best_model_path = self.checkpoint_dir.parent / "best_model.pth"
        best_info_path = self.checkpoint_dir.parent / "best_model_info.json"
        
        # Check if this is the best model so far
        should_save = True
        if best_info_path.exists():
            with open(best_info_path, 'r') as f:
                best_info = json.load(f)
            
            old_metric = best_info.get(metric_name, float('inf') if metric_name == 'loss' else float('-inf'))
            should_save = is_better(metric_value, old_metric)
        
        if should_save:
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                metric_name: metric_value,
                'timestamp': datetime.now().isoformat()
            }, best_model_path)
            
            # Save best model info
            best_info = {
                'epoch': epoch,
                metric_name: metric_value,
                'timestamp': datetime.now().isoformat()
            }
            with open(best_info_path, 'w') as f:
                json.dump(best_info, f, indent=2)
            
            print(f"🏆 New best model saved! {metric_name}={metric_value:.4f}")
            return True
        
        return False
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent ones."""
        if self.keep_last <= 0:
            return
        
        # Sort checkpoints by epoch
        self.saved_checkpoints.sort(key=lambda x: x['epoch'])
        
        # Remove old checkpoints
        while len(self.saved_checkpoints) > self.keep_last:
            old_checkpoint = self.saved_checkpoints.pop(0)
            if old_checkpoint['path'].exists():
                old_checkpoint['path'].unlink()
                print(f"🗑️ Removed old checkpoint: {old_checkpoint['path'].name}")
    
    def cleanup_all_checkpoints(self):
        """Remove all checkpoints."""
        for checkpoint_file in self.checkpoint_dir.glob("*.pth"):
            checkpoint_file.unlink()
        self.saved_checkpoints.clear()
        print(f"🗑️ Cleaned up all checkpoints in {self.checkpoint_dir}")


def save_model_for_inference(model: torch.nn.Module, 
                            save_path: Union[str, Path],
                            config: Optional[Dict[str, Any]] = None,
                            metadata: Optional[Dict[str, Any]] = None):
    """
    Save model for inference (without optimizer states).
    
    Args:
        model: Model to save
        save_path: Path to save the model
        config: Model configuration
        metadata: Additional metadata
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_data = {
        'model_state_dict': model.state_dict(),
        'timestamp': datetime.now().isoformat(),
        'config': config or {},
        'metadata': metadata or {}
    }
    
    torch.save(save_data, save_path)
    print(f"💾 Model saved for inference: {save_path}")


def load_model_for_inference(model: torch.nn.Module, 
                           checkpoint_path: Union[str, Path],
                           device: str = 'cpu') -> Dict[str, Any]:
    """
    Load model for inference.
    
    Args:
        model: Model instance to load weights into
        checkpoint_path: Path to checkpoint
        device: Device to load model on
    
    Returns:
        Loaded configuration and metadata
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"📂 Model loaded for inference: {checkpoint_path}")
    
    return {
        'config': checkpoint.get('config', {}),
        'metadata': checkpoint.get('metadata', {}),
        'timestamp': checkpoint.get('timestamp', 'unknown')
    } 