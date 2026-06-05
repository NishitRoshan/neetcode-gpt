import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        stats = []
        
        with torch.no_grad():
            out = x
            for layer in model.children():
                out = layer(out)
                
                if isinstance(layer, nn.Linear):
                    mean_val = out.mean().item()
                    std_val = out.std().item()
                    
                    # A neuron is dead if its output is <= 0 for ALL samples in the batch (dim 0)
                    is_dead = (out <= 0).all(dim=0)
                    dead_fraction = is_dead.float().mean().item()
                    
                    stats.append({
                        'mean': round(mean_val, 4),
                        'std': round(std_val, 4),
                        'dead_fraction': round(dead_fraction, 4)
                    })
                    
        return stats
        pass

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()
        
        criterion = nn.MSELoss()
        predictions = model(x)
        loss = criterion(predictions, y)
        loss.backward()
        
        stats = []
        for layer in model.children():
            if isinstance(layer, nn.Linear):
                grad = layer.weight.grad
                
                mean_val = grad.mean().item()
                std_val = grad.std().item()
                norm_val = torch.norm(grad).item()
                
                stats.append({
                    'mean': round(mean_val, 4),
                    'std': round(std_val, 4),
                    'norm': round(norm_val, 4)
                })
                
        return stats
        pass

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        for stat in activation_stats:
            if stat['dead_fraction'] > 0.5:
                return 'dead_neurons'
                
        # 2. Check for exploding gradients (norm > 1000)
        for stat in gradient_stats:
            if stat['norm'] > 1000:
                return 'exploding_gradients'
                
        # 3. Check for vanishing gradients in the last layer (norm < 1e-5)
        if gradient_stats and gradient_stats[-1]['norm'] < 1e-5:
            return 'vanishing_gradients'
            
        # 4. Check activation std for all layers
        for stat in activation_stats:
            if stat['std'] < 0.1:
                return 'vanishing_gradients'
            if stat['std'] > 10.0:
                return 'exploding_gradients'
                
        # 5. None of the above
        return 'healthy'
        pass
