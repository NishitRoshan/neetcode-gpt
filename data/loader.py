import torch
from typing import Tuple

class Solution:
    def create_batches(self, data: torch.Tensor, context_length: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(0)
        
        # Calculate the maximum valid starting index
        max_start_idx = len(data) - context_length
        
        # Generate random start indices for the batch
        start_indices = torch.randint(low=0, high=max_start_idx, size=(batch_size,))
        
        # Extract the slices and stack them into 2D tensors
        x = torch.stack([data[i : i + context_length] for i in start_indices])
        y = torch.stack([data[i + 1 : i + 1 + context_length] for i in start_indices])
        
        # This line is critical! It must return the tuple so it can be unpacked.
        return x, y
        pass