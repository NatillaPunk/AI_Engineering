import torch
import torch.nn as nn


# ACTIVATION ------------------------------------------------------------------------------------------------------------

relu = nn.ReLU() # ReLU (Rectified Linear Unit f(x)=max(0,x)): Zeros out negative values while keeping positive values
gelu  = nn.GELU() # GELU (Gaussian Error Linear Unit): Scales inputs based on how much greater they are than other inputs, using a smooth approximation.
swish = nn.SiLU() # Swish (Sigmoid Linear Unit f(x)=x⋅σ(x)): Multiplies the input by its sigmoid activation, allowing negative outputs when x is negative.









# NORMALIZATION ------------------------------------------------------------------------------------------------------------
# nn.BatchNorm1d(hidden_size) 
# BatchNorm: Each feature is normalized using its mean and variance computed over the entire batch.

# Commonly used in feedforward MLPs and CNNs
# Works best with reasonably large batch sizes
# Small batches may result in noisy estimates


# nn.LayerNorm(hidden_size)
# LayerNorm: Each row is normalized by computing the mean and variance of all features in that row.

# Commonly used in RNNs and transformers
# Works consistently regardless of batch size since it doesn’t depend on batch statistics
# Slightly more expensive computationally due to per-sample calculations