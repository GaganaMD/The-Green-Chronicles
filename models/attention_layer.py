import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(input_dim, 1)

    def forward(self, embeddings):
        # Calculate attention scores and normalize them across the sequence length
        scores = self.attention(embeddings)  # Shape: (batch_size, seq_length, 1)
        scores = F.softmax(scores, dim=1)
        # Compute the context vector as the weighted sum of embeddings
        context_vector = torch.sum(scores * embeddings, dim=1)  # Weighted sum across sequence
        return context_vector, scores
