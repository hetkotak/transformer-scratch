"""
Scaled Dot-Product Attention Implementation.

This module implements the core attention mechanism used in transformers:
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

This implementation uses self-attention where Q, K, and V are all derived from 
the same input through learned linear projections. It includes causal masking
to prevent the model from attending to future tokens (essential for GPT-style
autoregressive models).

Mathematical formulation:
1. Q = xW_q, K = xW_k, V = xW_v (linear projections)
2. Attention_scores = QK^T / sqrt(d_k) (scaled dot-product)
3. Apply causal mask to prevent future token attention
4. Attention_weights = softmax(Attention_scores) (normalization)
5. Output = Attention_weights @ V (weighted aggregation)

References:
- "Attention Is All You Need" (Vaswani et al., 2017)
- https://arxiv.org/abs/1706.03762
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism with causal masking.
    
    This is the core attention function used in transformers. It computes
    attention weights between queries and keys, then uses these weights
    to aggregate values. This implementation includes causal masking for
    autoregressive (GPT-style) models.
    
    Args:
        d_in (int): Input dimension (feature size of input embeddings)
        d_out (int): Output dimension (feature size of Q, K, V projections)
        context_length (int): Maximum sequence length for causal mask
        dropout_rate (float): Dropout rate for attention weights. Default: 0.1
        qkv_bias (bool): Whether to include bias in Q, K, V projections. Default: False
        
    Shape:
        - x: (batch_size, seq_len, d_in)
        - output: (batch_size, seq_len, d_out)
    """
    
    def __init__(self, d_in, d_out, context_length, dropout_rate: float = 0.1, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)  # Initializing linear layer of query weights - trainable
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias) # Initializing linear layer of key weights - trainable
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias) # Initializing linear layer of value weights - trainiable
        self.dropout = nn.Dropout(dropout_rate) # Adding a dropout layer to avoid overfitting
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )  # Adding mask to enable masking of future tokens
        
    def forward(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of scaled dot-product attention with causal masking.
        
        This method performs self-attention by:
        1. Computing Q, K, V from the same input x using learned linear projections
        2. Calculating attention scores: QK^T / sqrt(d_k)
        3. Applying causal mask to prevent attention to future tokens
        4. Computing attention weights via softmax
        5. Applying dropout to attention weights
        6. Computing final output: attention_weights @ V
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_in)
            
        Returns:
            context_vector: Attention output of shape (batch_size, seq_len, d_out)
        """
        batches, num_tokens, d_in = x.shape  # Extract dimensions from input tensor
        
        # Step 1: Generate Q, K, V from input using learned linear projections
        keys = self.W_key(x)     # K = xW_k, shape: (batch_size, seq_len, d_out)
        queries = self.W_query(x) # Q = xW_q, shape: (batch_size, seq_len, d_out)
        values = self.W_value(x)  # V = xW_v, shape: (batch_size, seq_len, d_out)

        # Step 2: Compute attention scores: QK^T
        attention_scores = queries @ keys.transpose(1,2) # Shape: (batch_size, seq_len, seq_len)
        
        # Step 3: Apply causal mask to prevent attention to future tokens
        attention_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        
        # Step 4: Scale by sqrt(d_k) and apply softmax for normalization
        attention_weights = torch.softmax(attention_scores / keys.shape[-1] ** 0.5, dim=-1)
        
        # Step 5: Apply dropout to attention weights for regularization
        attention_weights = self.dropout(attention_weights)
        
        # Step 6: Compute final output: weighted sum of values
        context_vector = attention_weights @ values # Shape: (batch_size, seq_len, d_out)
        return context_vector