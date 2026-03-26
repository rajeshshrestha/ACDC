from utils.stattools import normcdf
import torch

def loglikelihood_quantization(x, sigma, b_up, b_low):
    up_diff, low_diff = b_up - x, b_low -x
    up_diff, low_diff = up_diff/sigma, low_diff/sigma
    
    cdf_diff = torch.clip(normcdf(up_diff) - normcdf(low_diff), min=1e-30)
    loss = - torch.log(cdf_diff).sum(-1)
    
    return loss
    