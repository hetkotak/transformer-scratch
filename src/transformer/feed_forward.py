"""
Feed-forward networks implementation for Transformer architectures.

This module implements the position-wise feed-forward network (FFN) component
used in Transformer blocks. The FFN processes each position in the sequence
independently and identically, providing non-linearity and feature transformation.
"""

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN) for Transformer architectures.
    
    The FFN is a crucial component of each Transformer layer that processes
    the output from the multi-head attention mechanism. It applies the same
    fully connected feed-forward network to each position separately and identically.
    
    Architecture:
    - First linear layer: expands from d_model to 4 * d_model (expansion factor of 4)
    - Activation function: GELU (Gaussian Error Linear Unit)
    - Second linear layer: compresses back from 4 * d_model to d_model
    
    Mathematical representation:
    FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
    
    Where:
    - x: input tensor of shape (batch_size, seq_len, d_model)
    - W₁: first weight matrix of shape (d_model, 4 * d_model)
    - W₂: second weight matrix of shape (4 * d_model, d_model)
    - b₁, b₂: bias vectors
    
    Key Educational Points:
    1. **Position-wise**: The same FFN is applied to each token position independently
    2. **Expansion-Compression**: Temporarily expands to a larger dimension (4x) for
       more representational capacity, then compresses back to original size
    3. **GELU Activation**: Uses GELU instead of ReLU for smoother gradients and
       better performance (common in modern transformers like GPT)
    4. **Residual Connection**: This FFN is typically wrapped with a residual
       connection in the full transformer block
    
    References:
    - "Attention Is All You Need" (Vaswani et al., 2017) - Original Transformer paper
    - "Gaussian Error Linear Units (GELUs)" (Hendrycks & Gimpel, 2016)
    - https://arxiv.org/abs/1706.03762
    - https://arxiv.org/abs/1606.08415
    
    Args:
        cfg (dict): Configuration dictionary containing:
            - emb_dim (int): The embedding dimension (d_model in literature)
                           This is both input and output dimension
    """
    
    def __init__(self, cfg):
        """
        Initialize the Feed-Forward Network.
        
        Creates a sequential network with two linear transformations and GELU activation.
        The network follows the common transformer FFN pattern of expand → activate → compress.
        
        Educational Note:
        - The 4x expansion factor (emb_dim → 4 * emb_dim → emb_dim) is a standard
          choice in transformer architectures, providing sufficient representational
          capacity while maintaining computational efficiency.
        - GELU activation is preferred over ReLU in modern transformers as it provides
          smoother gradients and often better performance.
        
        Args:
            cfg (dict): Configuration dictionary containing:
                - emb_dim (int): Embedding dimension (also called d_model)
                               Both input and output will have this dimension
        
        Network Architecture:
        1. Linear layer: emb_dim → 4 * emb_dim (expansion)
        2. GELU activation: applies smooth non-linearity
        3. Linear layer: 4 * emb_dim → emb_dim (compression)
        """
        super().__init__()
        
        # Extract embedding dimension from config for clarity
        emb_dim = cfg["emb_dim"]
        
        # Create the feed-forward network as a sequential module
        # This makes the forward pass clear and matches the mathematical definition
        self.layers = nn.Sequential(
            # First linear transformation: expand to 4x the embedding dimension
            # This gives the network more representational capacity
            nn.Linear(emb_dim, 4 * emb_dim),
            
            # GELU activation function
            # GELU(x) = x * Φ(x) where Φ is the standard Gaussian CDF
            # Provides smooth, non-monotonic activation with better gradient flow
            nn.GELU(),
            
            # Second linear transformation: compress back to original dimension
            # This projection ensures the output can be added to the residual connection
            nn.Linear(4 * emb_dim, emb_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the feed-forward network to the input tensor.
        
        This method performs the core FFN computation: expand → activate → compress.
        The operation is applied identically to each position in the sequence,
        maintaining the sequence structure while transforming the feature representations.
        
        Educational Details:
        - **Position-wise**: Each token position is processed independently
        - **Identical processing**: The same weights are applied to all positions
        - **Feature transformation**: Maps from embedding space through a larger
          intermediate space and back, allowing complex feature interactions
        - **Maintains shape**: Input and output have the same shape for residual connections
        
        Mathematical operation:
        output = GELU(x @ W₁ + b₁) @ W₂ + b₂
        
        Where:
        - x: input tensor
        - W₁, W₂: learned weight matrices
        - b₁, b₂: learned bias vectors
        - @: matrix multiplication
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, emb_dim)
                            - batch_size: Number of sequences being processed
                            - seq_len: Length of each sequence (number of tokens)
                            - emb_dim: Embedding dimension (feature dimension per token)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, emb_dim)
                         Same shape as input, enabling residual connections
                         
        Example:
            >>> ffn = FeedForward({"emb_dim": 512})
            >>> x = torch.randn(32, 100, 512)  # batch=32, seq_len=100, emb_dim=512
            >>> output = ffn(x)  # Shape: (32, 100, 512)
        """
        # Apply the sequential layers: Linear → GELU → Linear
        # Each layer processes all positions simultaneously while maintaining
        # the batch and sequence dimensions
        return self.layers(x)