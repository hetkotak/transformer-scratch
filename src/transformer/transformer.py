"""
Transformer block implementation.
"""

import torch
import torch.nn as nn

from multihead_attention import MultiHeadAttention
from feed_forward import FeedForward
from layer_norm import LayerNorm

class TransformerBlock(nn.Module):
    """
    A single Transformer decoder block implementing the core building block of GPT models.
    
    This class combines multi-head self-attention, feed-forward network, layer normalization,
    and residual connections into a single transformer block. This implementation follows the
    "pre-normalization" variant (also known as "Pre-LN") where layer normalization is applied
    BEFORE the multi-head attention and feed-forward operations, rather than after.
    
    The pre-normalization approach has been shown to provide better training stability and
    performance compared to the original post-normalization design from the original
    Transformer paper.
    
    Architecture Flow:
    1. Input → LayerNorm → Multi-Head Self-Attention → Dropout → Add residual connection
    2. Output from step 1 → LayerNorm → Feed-Forward Network → Dropout → Add residual connection
    
    Why Pre-Normalization?
    - Improved gradient flow during training
    - Better training stability, especially for deeper models
    - Reduced risk of vanishing gradients
    - Commonly used in modern transformer implementations (GPT, BERT variants)
    
    Key Components:
    - Multi-Head Self-Attention: Allows the model to attend to different positions
    - Feed-Forward Network: Applies point-wise transformations to each position
    - Layer Normalization: Normalizes activations to stabilize training
    - Residual Connections: Enable gradient flow and help with deep network training
    - Dropout: Regularization to prevent overfitting
    
    References:
    - "Attention Is All You Need" (Vaswani et al., 2017) - https://arxiv.org/abs/1706.03762
    - "On Layer Normalization in the Transformer Architecture" (Xiong et al., 2020)
    - GPT-2 and GPT-3 implementations use this pre-norm architecture
    
    Args:
        cfg (dict): Configuration dictionary containing:
            - emb_dim (int): The embedding dimension / model dimension
            - n_heads (int): Number of attention heads
            - drop_rate (float): Dropout probability for regularization
            - context_length (int): Maximum sequence length for causal attention mask
            - qkv_bias (bool): Whether to use bias in query, key, value projections
    """
    
    def __init__(self, cfg):
        """
        Initialize the Transformer block with all necessary components.
        
        This constructor sets up:
        1. Multi-head self-attention mechanism for modeling dependencies between tokens
        2. Feed-forward network for applying non-linear transformations
        3. Two layer normalization modules (one before attention, one before FFN)
        4. Dropout layer for regularization of residual connections
        
        Args:
            cfg (dict): Configuration dictionary containing model hyperparameters:
                - emb_dim (int): Embedding/model dimension (typically 512, 768, 1024, etc.)
                - n_heads (int): Number of attention heads (emb_dim must be divisible by n_heads)
                - drop_rate (float): Dropout probability (typically 0.1 for training, 0.0 for inference)
                - context_length (int): Maximum sequence length the model can handle
                - qkv_bias (bool): Whether to include bias terms in Q, K, V linear projections
        """
        super().__init__()
        
        # Multi-head self-attention mechanism
        # This allows the model to jointly attend to information from different 
        # representation subspaces at different positions
        self.attention = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            dropout=cfg["drop_rate"],
            num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"]
        )
        
        # Feed-forward network for point-wise transformations
        # Applies the same fully connected network to each position separately
        self.ff = FeedForward(cfg=cfg)
        
        # Layer normalization modules for stabilizing training
        # norm1: Applied before the attention mechanism (pre-norm)
        # norm2: Applied before the feed-forward network (pre-norm)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        
        # Dropout for regularizing the residual connections
        # Applied to outputs of attention and feed-forward before adding to residual
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transformer block using pre-normalization architecture.
        
        This method implements the core transformer computation with two main sub-blocks:
        1. Multi-head self-attention sub-block with residual connection
        2. Feed-forward network sub-block with residual connection
        
        Pre-normalization Flow:
        For each sub-block: Input → LayerNorm → Transformation → Dropout → Add to Input
        
        This differs from post-normalization where the flow would be:
        Input → Transformation → Dropout → Add to Input → LayerNorm
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, emb_dim)
                            - batch_size: Number of sequences in the batch
                            - seq_len: Length of each sequence (number of tokens)
                            - emb_dim: Embedding dimension (model dimension)
            
        Returns:
            torch.Tensor: Output tensor of same shape (batch_size, seq_len, emb_dim)
                         Contains contextually-aware representations of input tokens
        """
        # ===== MULTI-HEAD SELF-ATTENTION SUB-BLOCK =====
        # Store the input for the residual connection
        # This preserves the original signal and helps with gradient flow
        shortcut = x
        
        # Apply layer normalization BEFORE the attention operation (pre-norm)
        # This normalizes the input to have zero mean and unit variance
        # Helps stabilize training and improve convergence
        x = self.norm1(x)
        
        # Apply multi-head self-attention
        # This allows each token to attend to all previous tokens (causal attention)
        # The attention mechanism computes weighted combinations of all token representations
        x = self.attention(x)
        
        # Apply dropout to the attention output for regularization
        # This randomly sets some elements to zero during training to prevent overfitting
        x = self.drop_shortcut(x)
        
        # Add the residual connection (skip connection)
        # This allows gradients to flow directly through the network and helps with training deep models
        # Formula: output = attention_output + original_input
        x = x + shortcut
        
        # ===== FEED-FORWARD NETWORK SUB-BLOCK =====
        # Store the output of attention block for the next residual connection
        shortcut = x
        
        # Apply layer normalization BEFORE the feed-forward network (pre-norm)
        # Again, this helps with training stability
        x = self.norm2(x)
        
        # Apply feed-forward network
        # This applies the same fully connected network to each position independently
        # Typically: Linear → ReLU/GELU → Linear with expansion factor (usually 4x)
        x = self.ff(x)
        
        # Apply dropout to the feed-forward output
        x = self.drop_shortcut(x)
        
        # Add the second residual connection
        # This connects the input to the FFN sub-block with its output
        x = x + shortcut
        
        return x
