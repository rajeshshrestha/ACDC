import torch
import math

def normcdf(x):
    """
    Compute the cumulative distribution function (CDF) for a normal distribution.
    
    Args:
    x (torch.Tensor): Input tensor.
    
    Returns:
    torch.Tensor: CDF values.
    """
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

def normpdf(x):
    """
    Compute the probability density function (PDF) for a normal distribution.
    
    Args:
    x (torch.Tensor): Input tensor.
    
    Returns:
    torch.Tensor: PDF values.
    """
    return (1 / math.sqrt(2 * math.pi)) * torch.exp(-0.5 * x ** 2)