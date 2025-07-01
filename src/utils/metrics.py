"""
Layout Metrics for Diffusion Section Transformer
Basic metrics computation for layout accuracy and quality assessment.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Union
from PIL import Image
import cv2


class LayoutMetrics:
    """Basic layout metrics computation."""
    
    def __init__(self):
        self.metric_names = [
            'layout_accuracy',
            'element_precision', 
            'element_recall',
            'element_f1',
            'token_accuracy'
        ]
    
    def compute_batch_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute metrics for a batch of predictions and targets.
        
        Args:
            predictions: Model predictions [batch_size, seq_len, ...] (may have embeddings)
            targets: Target tokens [batch_size, seq_len] (typically discrete)
            
        Returns:
            Dictionary of metrics
        """
        batch_size = predictions.size(0)
        
        # Handle different tensor shapes
        pred_tokens = predictions
        target_tokens = targets
        
        # If predictions have extra dimensions (embeddings), reduce them
        if pred_tokens.dim() > 2:
            # Average over the embedding dimension to get sequence-level predictions
            pred_tokens = pred_tokens.mean(dim=-1)
        
        # If targets have extra dimensions, flatten appropriately
        if target_tokens.dim() > 2:
            target_tokens = target_tokens.view(batch_size, -1)
        
        # Handle sequence length mismatch
        pred_seq_len = pred_tokens.size(1)
        target_seq_len = target_tokens.size(1)
        
        if pred_seq_len != target_seq_len:
            # Align sequence lengths by taking the minimum and padding/truncating
            min_seq_len = min(pred_seq_len, target_seq_len)
            
            if pred_seq_len > min_seq_len:
                pred_tokens = pred_tokens[:, :min_seq_len]
            elif pred_seq_len < min_seq_len:
                # Pad predictions with zeros
                padding = torch.zeros(batch_size, min_seq_len - pred_seq_len, device=pred_tokens.device)
                pred_tokens = torch.cat([pred_tokens, padding], dim=1)
            
            if target_seq_len > min_seq_len:
                target_tokens = target_tokens[:, :min_seq_len]
            elif target_seq_len < min_seq_len:
                # Pad targets with zeros
                padding = torch.zeros(batch_size, min_seq_len - target_seq_len, device=target_tokens.device)
                target_tokens = torch.cat([target_tokens, padding], dim=1)
        
        # Now compute metrics with aligned tensors
        try:
            # Token-level accuracy (handle continuous vs discrete comparison)
            if target_tokens.max() > 10 and torch.all(target_tokens == target_tokens.int()):
                # Discrete targets - convert predictions to discrete for comparison
                pred_discrete = torch.round(torch.sigmoid(pred_tokens) * target_tokens.max())
                correct_tokens = (pred_discrete == target_tokens).float()
            else:
                # Continuous comparison
                diff = torch.abs(pred_tokens - target_tokens)
                threshold = 0.1  # Tolerance for "correct" predictions
                correct_tokens = (diff < threshold).float()
            
            token_accuracy = correct_tokens.mean().item()
        except Exception as e:
            print(f"⚠️ Error computing token accuracy: {e}")
            token_accuracy = 0.0
        
        # Layout accuracy - use improved computation
        layout_accuracy = self.compute_layout_accuracy(predictions, targets)
        
        # Element-level metrics (simplified)
        element_precision = []
        element_recall = []
        element_f1 = []
        
        for i in range(batch_size):
            pred_set = set(pred_tokens[i].cpu().numpy())
            target_set = set(target_tokens[i].cpu().numpy())
            
            if len(pred_set) > 0:
                precision = len(pred_set & target_set) / len(pred_set)
            else:
                precision = 0.0
            
            if len(target_set) > 0:
                recall = len(pred_set & target_set) / len(target_set)
            else:
                recall = 0.0
            
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
            
            element_precision.append(precision)
            element_recall.append(recall)
            element_f1.append(f1)
        
        # Aggregate metrics
        metrics = {
            'layout_accuracy': layout_accuracy,
            'element_precision': element_precision,
            'element_recall': element_recall,
            'element_f1': element_f1,
            'token_accuracy': token_accuracy
        }
        
        return metrics
    
    def compute_comprehensive_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
        """Compute comprehensive metrics including statistical measures."""
        batch_metrics = self.compute_batch_metrics(predictions, targets)
        
        # Convert lists to statistics
        comprehensive = {}
        for key, values in batch_metrics.items():
            if isinstance(values, list):
                comprehensive[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
            else:
                comprehensive[key] = {'mean': values, 'std': 0.0, 'min': values, 'max': values}
        
        return comprehensive

    def compute_layout_accuracy(self, predicted_tokens: torch.Tensor, target_tokens: torch.Tensor) -> List[float]:
        """
        Compute layout accuracy with improved handling for continuous vs discrete tokens.
        
        Args:
            predicted_tokens: Model predictions [batch_size, seq_len, ...] (may be continuous)
            target_tokens: Target tokens [batch_size, seq_len] (typically discrete)
            
        Returns:
            List of accuracy scores per sample
        """
        batch_size = predicted_tokens.size(0)
        accuracy_scores = []
        
        for i in range(batch_size):
            pred = predicted_tokens[i]
            target = target_tokens[i]
            
            try:
                # Handle different dimensionalities
                if pred.dim() > target.dim():
                    # If pred has extra dimensions (e.g., embeddings), use mean pooling
                    pred = pred.mean(dim=-1) if pred.dim() > 1 else pred
                elif pred.dim() < target.dim():
                    # If target has extra dimensions, flatten it
                    target = target.view(-1) if target.dim() > 1 else target
                
                # Ensure same sequence length
                min_len = min(pred.size(0), target.size(0))
                pred = pred[:min_len]
                target = target[:min_len].float()  # Convert to float for comparison
                
                # Check if we have discrete tokens (likely integers)
                if target.max() > 10 and target.min() >= 0 and torch.all(target == target.int()):
                    # Target appears to be discrete tokens - use similarity-based matching
                    # Normalize predictions to be in a reasonable range
                    pred_normalized = torch.sigmoid(pred)  # Map to [0, 1]
                    target_normalized = target / (target.max() + 1e-8)  # Normalize discrete tokens
                    
                    # Compute cosine similarity
                    similarity = F.cosine_similarity(pred_normalized.unsqueeze(0), target_normalized.unsqueeze(0))
                    accuracy = similarity.item()
                else:
                    # Both are continuous - use correlation-based accuracy
                    if torch.std(pred) > 1e-8 and torch.std(target) > 1e-8:
                        # Compute correlation coefficient
                        pred_centered = pred - pred.mean()
                        target_centered = target - target.mean()
                        correlation = (pred_centered * target_centered).sum() / (torch.sqrt((pred_centered**2).sum()) * torch.sqrt((target_centered**2).sum()))
                        accuracy = max(0.0, correlation.item())  # Clamp to [0, 1]
                    else:
                        # One of them is constant - use MSE-based accuracy
                        mse = F.mse_loss(pred, target)
                        accuracy = 1.0 / (1.0 + mse.item())  # Convert MSE to accuracy-like metric
                
                accuracy_scores.append(min(1.0, max(0.0, accuracy)))  # Clamp to [0, 1]
                
            except Exception as e:
                # Fallback for any errors
                accuracy_scores.append(0.0)
        
        return accuracy_scores


class VisualSimilarityMetrics:
    """Metrics for visual similarity assessment."""
    
    def __init__(self):
        pass
    
    def compute_similarity_metrics(self, generated_images: List[Image.Image], 
                                 target_images: List[Image.Image]) -> Dict[str, float]:
        """
        Compute visual similarity metrics between generated and target images.
        
        Args:
            generated_images: List of generated layout images
            target_images: List of target layout images
        
        Returns:
            Dictionary of visual similarity metrics
        """
        if len(generated_images) != len(target_images):
            raise ValueError("Number of generated and target images must match")
        
        ssim_scores = []
        mse_scores = []
        
        for gen_img, target_img in zip(generated_images, target_images):
            # Convert to numpy arrays
            gen_array = np.array(gen_img.convert('RGB'))
            target_array = np.array(target_img.convert('RGB'))
            
            # Resize if needed
            if gen_array.shape != target_array.shape:
                gen_array = cv2.resize(gen_array, (target_array.shape[1], target_array.shape[0]))
            
            # Compute MSE
            mse = np.mean((gen_array - target_array) ** 2)
            mse_scores.append(mse)
            
            # Compute simplified SSIM-like metric
            # (This is a basic version - full SSIM would require more computation)
            gen_gray = cv2.cvtColor(gen_array, cv2.COLOR_RGB2GRAY)
            target_gray = cv2.cvtColor(target_array, cv2.COLOR_RGB2GRAY)
            
            # Simplified structural similarity
            correlation = np.corrcoef(gen_gray.flatten(), target_gray.flatten())[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            ssim_scores.append(max(0.0, correlation))
        
        return {
            'visual_similarity_ssim': np.mean(ssim_scores),
            'visual_similarity_mse': np.mean(mse_scores),
            'visual_similarity_psnr': 20 * np.log10(255.0 / (np.mean(mse_scores) + 1e-8))
        }


class AestheticMetrics:
    """Metrics for aesthetic quality assessment."""
    
    def __init__(self):
        pass
    
    def compute_aesthetic_scores(self, generated_layouts: torch.Tensor) -> Dict[str, float]:
        """
        Compute aesthetic quality scores for generated layouts.
        
        Args:
            generated_layouts: Generated layout representations
        
        Returns:
            Dictionary of aesthetic scores
        """
        batch_size = generated_layouts.size(0)
        
        # Basic aesthetic metrics (simplified)
        balance_scores = []
        consistency_scores = []
        
        for i in range(batch_size):
            layout = generated_layouts[i]
            
            # Balance score (variance in token distribution)
            token_variance = torch.var(layout.float()).item()
            balance_score = 1.0 / (1.0 + token_variance)  # Lower variance = better balance
            balance_scores.append(balance_score)
            
            # Consistency score (repetition patterns)
            unique_tokens = len(torch.unique(layout))
            total_tokens = len(layout)
            consistency_score = unique_tokens / max(total_tokens, 1)
            consistency_scores.append(consistency_score)
        
        return {
            'aesthetic_balance': np.mean(balance_scores),
            'aesthetic_consistency': np.mean(consistency_scores),
            'aesthetic_overall': np.mean([np.mean(balance_scores), np.mean(consistency_scores)])
        } 