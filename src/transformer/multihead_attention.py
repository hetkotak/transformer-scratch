"""
Multi-Head Attention Implementation.

This module implements multi-head attention, which applies multiple attention
heads in parallel and concatenates their outputs. Each head uses scaled
dot-product attention with different learned linear projections.

References:
- "Attention Is All You Need" (Vaswani et al., 2017)
- https://arxiv.org/abs/1706.03762
"""

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention Module for Transformer architectures.
    
    Multi-head attention allows the model to jointly attend to information from different
    representation subspaces at different positions. Instead of performing a single
    attention function with d_model-dimensional keys, values and queries, we linearly
    project the queries, keys and values h times with different, learned linear 
    projections to d_k, d_k and d_v dimensions, respectively.
    
    The multi-head attention mechanism:
    1. Projects input to Q, K, V using learned linear transformations
    2. Splits each into multiple heads (different representation subspaces)
    3. Applies scaled dot-product attention to each head in parallel
    4. Concatenates the head outputs
    5. Applies a final linear projection
    
    This implementation uses causal (autoregressive) masking to prevent tokens
    from attending to future positions, making it suitable for language modeling.
    
    Args in forward():
        x: Input tensor of shape (batch_size, num_tokens, d_in)
        
    Returns:
        Output tensor of shape (batch_size, num_tokens, d_out)
    """
    
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        """
        Initialize the Multi-Head Attention module.
        
        Args:
            d_in (int): Input embedding dimension (size of input feature vectors)
            d_out (int): Output embedding dimension (size of output feature vectors)
                        Must be divisible by num_heads
            context_length (int): Maximum sequence length for causal masking
            dropout (float): Dropout probability (0.0 to 1.0) for regularization
            num_heads (int): Number of parallel attention heads
            qkv_bias (bool, optional): Whether to include bias terms in Q, K, V 
                                     linear projections. Defaults to False.
        
        Note:
            Each attention head will have dimension d_out // num_heads.
            The final output will have the same dimension as d_out.
        """
        super().__init__()
        
        # Ensure d_out can be evenly divided among all heads
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        # Store configuration parameters
        self.d_out = d_out
        self.num_heads = num_heads
        # Each head processes a smaller dimension to maintain computational efficiency
        # while allowing different heads to focus on different representation subspaces
        self.head_dim = d_out // num_heads
        
        # Linear transformations to create queries, keys, and values for all heads
        # These projections are learned during training and allow the model to
        # transform input embeddings into appropriate representations for attention
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)  # Query projection
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)    # Key projection  
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)  # Value projection
        
        # Final linear layer to combine and transform concatenated head outputs
        # This allows the model to learn optimal combinations of different head representations
        self.out_proj = nn.Linear(d_out, d_out)
        
        # Dropout layer for regularization (prevents overfitting)
        self.dropout = nn.Dropout(dropout)
        
        # Create and register a causal mask as a buffer (not a learnable parameter)
        # This mask prevents tokens from attending to future tokens (causal/autoregressive)
        # register_buffer ensures this tensor moves with the model to GPU/CPU
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        """
        Forward pass of multi-head attention.
        
        Performs the complete multi-head attention computation:
        1. Projects input to queries, keys, and values
        2. Splits projections into multiple attention heads
        3. Computes scaled dot-product attention for each head
        4. Applies causal masking to prevent future token access
        5. Concatenates head outputs and applies final projection
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_tokens, d_in)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_tokens, d_out)
        """
        # Extract dimensions from input tensor
        batch_size, num_tokens, d_in = x.shape
        
        # Step 1: Create queries, keys, and values by applying learned linear transformations
        # Each projection has shape (batch_size, num_tokens, d_out)
        keys = self.W_key(x)      # Transform input to keys for similarity computation
        queries = self.W_query(x) # Transform input to queries for what to look for
        values = self.W_value(x)  # Transform input to values for what information to extract

        # Step 2: Reshape tensors to separate different attention heads
        # We split the d_out dimension into (num_heads, head_dim) where head_dim = d_out / num_heads
        # Then transpose to put heads dimension second for efficient parallel processing
        # Final shape: (batch_size, num_heads, num_tokens, head_dim)
        
        # Reshape: (batch_size, num_tokens, d_out) -> (batch_size, num_tokens, num_heads, head_dim)
        # Then transpose: (batch_size, num_tokens, num_heads, head_dim) -> (batch_size, num_heads, num_tokens, head_dim)
        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1,2)
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1,2)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1,2)

        # Step 3: Compute attention scores using scaled dot-product attention
        # For each head: multiply queries with keys (transposed) to get similarity scores
        # Shape: (batch_size, num_heads, num_tokens, num_tokens)
        attention_scores = queries @ keys.transpose(2, 3)

        # Step 4: Apply causal mask to prevent attending to future tokens
        # Truncate the mask to match current sequence length (for efficiency)
        # Cast to bool and slice to current sequence length
        mask_bool = self.mask.bool()[:num_tokens,:num_tokens]
        # Fill masked positions with -infinity so they become 0 after softmax
        attention_scores.masked_fill_(mask_bool, -torch.inf)

        # Step 5: Convert attention scores to attention weights using softmax
        # Scale by sqrt(head_dim) to prevent extremely large values (as per "Attention Is All You Need" paper)
        # Softmax normalizes scores so they sum to 1 across the last dimension (each token's attention distribution)
        attention_weights = torch.softmax(attention_scores / self.head_dim**0.5, dim=-1)

        # Step 6: Apply dropout to attention weights for regularization
        attention_weights = self.dropout(attention_weights)

        # Step 7: Apply attention weights to values to get context vectors
        # This computes a weighted sum where attention weights determine how much each value contributes
        # Result shape: (batch_size, num_heads, num_tokens, head_dim)
        context_vector = (attention_weights @ values).transpose(1, 2)
        # Transpose back to: (batch_size, num_tokens, num_heads, head_dim)

        # Step 8: Concatenate all heads back together
        # Reshape from (batch_size, num_tokens, num_heads, head_dim) to (batch_size, num_tokens, d_out)
        # where d_out = num_heads * head_dim
        # .contiguous() ensures memory layout is contiguous for efficient view operation
        context_vector = context_vector.contiguous().view(batch_size, num_tokens, self.d_out)

        # Step 9: Apply final linear projection to combine information from all heads
        # This allows the model to learn how to optimally combine the different head outputs
        context_vector = self.out_proj(context_vector)

        return context_vector