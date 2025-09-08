"""
Layer Normalization for Transformer architectures implementation.

This module implements Layer Normalization (LayerNorm), a crucial component in modern
Transformer architectures that helps stabilize training and improve model performance.
"""

import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """
    Layer Normalization for Transformer architectures.
    
    WHAT IS LAYER NORMALIZATION?
    ============================
    Layer normalization is a technique that normalizes the inputs across the feature 
    dimension for each individual sample in a batch. This is different from batch 
    normalization, which normalizes across the batch dimension.
    
    WHY DO WE NEED IT?
    ==================
    1. **Training Stability**: Prevents internal covariate shift by ensuring inputs 
       to each layer have stable statistics (mean ≈ 0, variance ≈ 1)
    2. **Gradient Flow**: Helps gradients flow better during backpropagation
    3. **Batch Independence**: Unlike batch norm, works with any batch size (even 1)
    4. **Faster Convergence**: Often leads to faster training convergence
    
    HOW DOES IT WORK?
    =================
    For each sample in the batch, LayerNorm:
    1. Computes the mean (μ) and variance (σ²) across all features
    2. Normalizes the input: (x - μ) / √(σ² + ε)
    3. Applies learned scale (γ) and shift (β) parameters
    
    Mathematical Formula:
    LayerNorm(x) = γ ⊙ ((x - μ) / √(σ² + ε)) + β
    
    Where:
    - x: input tensor of shape (batch_size, seq_len, d_model)
    - μ: mean computed across the d_model dimension for each position
    - σ²: variance computed across the d_model dimension for each position
    - ε (eps): small constant for numerical stability (prevents division by zero)
    - γ (gamma/scale): learnable parameter that controls the scale of normalized output
    - β (beta/shift): learnable parameter that controls the shift of normalized output
    - ⊙: element-wise multiplication
    
    EXAMPLE:
    ========
    Input tensor shape: (2, 3, 4)  # batch_size=2, seq_len=3, d_model=4
    
    For each position (i, j), we normalize across the feature dimension:
    - x[i, j, :] has shape (4,) - these 4 values get normalized together
    - We compute mean and variance of these 4 values
    - Apply normalization and learnable transformation
    
    COMPARISON WITH BATCH NORMALIZATION:
    ===================================
    - Batch Norm: Normalizes across batch dimension (axis=0)
    - Layer Norm: Normalizes across feature dimension (axis=-1)
    - Layer Norm is more suitable for NLP tasks with variable sequence lengths
    
    References:
    - "Layer Normalization" (Ba et al., 2016): https://arxiv.org/abs/1607.06450
    - "Attention Is All You Need" (Vaswani et al., 2017): https://arxiv.org/abs/1706.03762
    
    Args:
        d_model (int): The dimension of the model features (embedding dimension)
        eps (float): Small value added to variance for numerical stability (default: 1e-5)
    """
    
    def __init__(self, d_model: int, eps: float = 1e-5):
        """
        Initialize the Layer Normalization module.
        
        This constructor sets up the learnable parameters and constants needed for
        layer normalization. The scale and shift parameters allow the model to learn
        the optimal normalization for each feature dimension.
        
        LEARNABLE PARAMETERS:
        ====================
        - scale (γ): Initialized to ones - allows the model to learn how much to 
          scale each normalized feature. If a feature should have larger magnitude
          after normalization, the model will learn a scale > 1 for that dimension.
        
        - shift (β): Initialized to zeros - allows the model to learn an offset for
          each feature after normalization. This gives the model flexibility to 
          shift the normalized distribution as needed.
        
        WHY LEARNABLE PARAMETERS?
        =========================
        Without γ and β, layer norm would always force the output to have mean=0 
        and variance=1. The learnable parameters give the model the flexibility to
        learn the optimal statistics for each layer while still benefiting from
        the normalization stability.
        
        Args:
            d_model (int): The dimension of the model features (e.g., 512, 768, etc.)
                          This determines the size of the scale and shift parameter vectors.
            eps (float): Small constant for numerical stability (default: 1e-5)
                        Added to variance before taking square root to prevent division by zero.
                        
        Example:
            # For a model with 512-dimensional embeddings
            layer_norm = LayerNorm(d_model=512)
            # This creates scale and shift parameters of size (512,)
        """
        super().__init__()
        
        # Store epsilon for numerical stability in forward pass
        self.eps = eps
        
        # γ (gamma) - learnable scale parameter
        # Shape: (d_model,) - one scale value per feature dimension
        # Initialized to 1.0 so initially the scaling is identity
        self.scale = nn.Parameter(torch.ones(d_model))
        
        # β (beta) - learnable shift parameter  
        # Shape: (d_model,) - one shift value per feature dimension
        # Initialized to 0.0 so initially there's no shift
        self.shift = nn.Parameter(torch.zeros(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization to the input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Calculate mean and variance across the feature dimension (last dimension)
        mean = x.mean(dim=-1, keepdim=True)

        # For LLMs, where the embedding dimension n is significantly large, the difference between using n and n – 1 is practically negligible.
        # Hence keeping unbiased=False to use n instead of n-1. This is tensorflow's default behavior which was used with GPT-2
        # This also ensures our model is compatible with pre-trained weights of GPT-2
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        # Scale and shift
        return self.scale * normalized + self.shift