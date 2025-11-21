import torch
import torch.nn as nn
import math

class SoftmaxDotProductApprox(nn.Module):
    """
    Approximates the unnormalized dot-product mat-vec K v = (Q K^T) v using Softmax Attention.
    
    Theory:
    Softmax(epsilon * Q K^T) v 
    = (1/Z) * sum_j exp(epsilon * q_i^T k_j) v_j
    ~ (1/Z) * sum_j (1 + epsilon * q_i^T k_j) v_j
    = (1/Z) * (sum_j v_j + epsilon * sum_j (q_i^T k_j) v_j)
    
    If we assume Z ~ N (valid for small epsilon and centered q/k),
    and we subtract the mean value (sum_j v_j) / N (using a uniform attention head),
    we isolate the dot product term.
    
    Approximation:
    (N / epsilon) * (Softmax(epsilon * Q K^T) v - Mean(v))
    """
    
    def __init__(self, d_model: int, epsilon: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.epsilon = epsilon
        
        # We don't strictly need learnable parameters for the constructive proof,
        # but we structure it as a module.
        
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Q: (batch, n_query, d)
            K: (batch, n_key, d)
            V: (batch, n_key, d_v)
        Returns:
            Approximate (Q @ K.T) @ V
        """
        n_key = K.shape[1]
        scale = self.epsilon
        
        # Head 1: Scaled Softmax Attention
        # scores = epsilon * Q K^T
        scores = torch.bmm(Q, K.transpose(1, 2)) * scale
        attn_probs = torch.softmax(scores, dim=-1) # (batch, n_query, n_key)
        head1 = torch.bmm(attn_probs, V) # (batch, n_query, d_v)
        
        # Head 2: Uniform Mean (Global Average of V)
        # Construct uniform attention weights (1/N)
        # Note: In a real Transformer, this corresponds to 0-temperature or zero inputs?
        # Actually, 0 inputs => exp(0)=1 => uniform weights.
        # So we can simulate this by passing Zero Q and Zero K to standard attention, or just computing mean.
        # For the "two-head" construction, we assume we have a head that attends uniformly.
        # sum_j v_j / N
        mean_v = V.mean(dim=1, keepdim=True) # (batch, 1, d_v)
        # Broadcast to n_query
        head2 = mean_v.expand_as(head1)
        
        # Combine: (Head1 - Head2) * (N / epsilon)
        # H1 ~ (1/N)(Sum V + eps * KV)
        # H2 ~ (1/N)(Sum V)
        # H1 - H2 ~ (eps/N) * KV
        # * (N/eps) ~ KV
        
        approx_kv = (head1 - head2) * (n_key / self.epsilon)
        
        return approx_kv

    def compute_exact(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return torch.bmm(torch.bmm(Q, K.transpose(1, 2)), V)

