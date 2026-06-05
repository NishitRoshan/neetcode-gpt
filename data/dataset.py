import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]], List[List[str]]]:
        tokens = raw_dataset.split()
        torch.manual_seed(0)
        
        max_start_idx = len(tokens) - context_length
        start_indices = torch.randint(low=0, high=max_start_idx, size=(batch_size,))
        X = []
        Y = []
        for idx in start_indices.tolist():
            X.append(tokens[idx : idx + context_length])
            Y.append(tokens[idx + 1 : idx + 1 + context_length])
        return X, Y