"""
GPT-2 Style Decoder-Only Transformer Model

This module implements a simplified version of the GPT-2 architecture, which is a 
decoder-only transformer model designed for autoregressive language generation.
The model predicts the next token in a sequence given the previous tokens.

Key architectural components:
- Token embeddings: Convert input token IDs to dense vector representations
- Positional embeddings: Add position information to each token
- Transformer blocks: Stack of self-attention and feed-forward layers
- Layer normalization: Stabilize training and improve convergence
- Output head: Project hidden states back to vocabulary space for next-token prediction
"""

import torch
import torch.nn as nn
from transformer import TransformerBlock
from layer_norm import LayerNorm

class GPTModel(nn.Module):
    """
    GPT-2 style decoder-only transformer model.
    
    This model follows the GPT architecture where each position can only attend to 
    previous positions (causal/masked self-attention), making it suitable for 
    autoregressive text generation.
    
    Args:
        cfg (dict): Configuration dictionary containing:
            - vocab_size (int): Size of the vocabulary (number of unique tokens)
            - emb_dim (int): Embedding dimension (hidden size)
            - context_length (int): Maximum sequence length the model can handle
            - drop_rate (float): Dropout probability for regularization
            - n_layers (int): Number of transformer blocks to stack
    """
    
    def __init__(self, cfg):
        super().__init__()
        
        # Token embeddings: Map each token ID to a dense vector of size emb_dim
        # This creates a learnable lookup table where each of the vocab_size tokens
        # gets its own emb_dim-dimensional vector representation
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        
        # Positional embeddings: Add position information to tokens
        # Since transformers have no inherent notion of sequence order, we need
        # to explicitly encode position information. This creates learnable
        # position vectors for each possible position (0 to context_length-1)
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        
        # Embedding dropout: Regularization applied after combining token and position embeddings
        # Helps prevent overfitting by randomly setting some embedding values to zero during training
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Stack of transformer blocks: The core of the model
        # Each TransformerBlock contains masked self-attention and feed-forward layers
        # with residual connections and layer normalization
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # Final layer normalization: Applied before the output projection
        # Helps stabilize the final hidden representations before converting to logits
        self.final_norm = LayerNorm(cfg["emb_dim"])
        
        # Output head: Projects hidden states back to vocabulary space
        # Linear layer that converts emb_dim features to vocab_size logits
        # No bias is used here following the original GPT-2 implementation
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        """
        Forward pass through the GPT model.
        
        This method implements the complete forward pass of the GPT-2 model:
        1. Convert input token IDs to embeddings
        2. Add positional information 
        3. Apply dropout for regularization
        4. Process through transformer blocks
        5. Apply final normalization
        6. Project to vocabulary space for next-token prediction
        
        Args:
            in_idx (torch.Tensor): Input token indices of shape (batch_size, seq_len)
                                  Each element should be an integer in range [0, vocab_size-1]
        
        Returns:
            torch.Tensor: Logits of shape (batch_size, seq_len, vocab_size)
                         Raw scores for each token in the vocabulary at each position.
                         Higher scores indicate higher probability for that token.
        """
        # Extract batch size and sequence length from input tensor
        batch_size, seq_len = in_idx.shape
        
        # Convert token IDs to dense vector representations
        # Shape: (batch_size, seq_len, emb_dim)
        tok_embeds = self.tok_emb(in_idx)
        
        # Create positional embeddings for the current sequence
        # torch.arange creates position indices [0, 1, 2, ..., seq_len-1]
        # pos_emb maps these to learned positional vectors
        # Shape: (seq_len, emb_dim)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        
        # Combine token and positional embeddings
        # Broadcasting allows adding (seq_len, emb_dim) to (batch_size, seq_len, emb_dim)
        # This gives each token both its semantic meaning and positional context
        # Shape: (batch_size, seq_len, emb_dim)
        x = tok_embeds + pos_embeds
        
        # Apply dropout to the combined embeddings for regularization
        # Only active during training (automatically disabled during eval)
        x = self.drop_emb(x)
        
        # Process through the stack of transformer blocks
        # Each block applies masked self-attention and feed-forward transformations
        # with residual connections and layer normalization
        # Shape remains: (batch_size, seq_len, emb_dim)
        x = self.trf_blocks(x)
        
        # Apply final layer normalization to stabilize the representations
        # Shape remains: (batch_size, seq_len, emb_dim)
        x = self.final_norm(x)
        
        # Project hidden states to vocabulary space to get next-token predictions
        # Each position gets a score for every possible next token
        # Shape: (batch_size, seq_len, vocab_size)
        logits = self.out_head(x)
        
        return logits
