import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value 
        self.attention_dim = attention_dim
        self.key = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value = nn.Linear(embedding_dim, attention_dim, bias=False)
        pass

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        batch_size, context_length, _ = embedded.shape
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        k = self.key(embedded)
        q = self.query(embedded)
        v = self.value(embedded)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.attention_dim)
        mask = torch.tril(torch.ones(context_length, context_length)).to(embedded.device)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = torch.softmax(scores, dim=2)
        output = attention_weights @ v
        return torch.round(output, decimals=4)
        pass
