from .registry import register_operator
from .base import Operator
import torch
from utils.quantization import quantize_uniform
from utils.losses import loglikelihood_quantization
from utils.stattools import normpdf, normcdf
import tqdm
import wandb as wnb

'''Compression and Quantization operator'''
@register_operator(name='compression_quantization')
class CompressionQuantization(Operator):
    def __init__(self, compression_factor, nbits,  device='cuda', sigma=0.05):
        super().__init__(sigma)
        self.compression_factor = compression_factor
        self.nbits = nbits
        self.A = None
        self.R = None
        self.input_dim = None
        self.device = device
        self.C_pinv = None

    def compress(self, x):
        return self.A @ x.contiguous().view(*x.shape[:-3], -1, 1)

    def dither(self, x):
        return x + self.sigma * torch.randn_like(x)

    def measure(self, x):
        return self(x)

    def __call__(self, x):
        # Populate in the first run
        if self.A is None or self.R is None or self.input_dim is None:
            if len(x.shape) == 3:
                self.input_dim = torch.numel(x)
            elif len(x.shape) == 4:
                self.input_dim = torch.numel(x[0])
            else:
                raise Exception(f"Invalid image shape provided: {x.shape}")

            # Number of measurement
            self.R = self.input_dim//self.compression_factor

            self.A = 1/self.R**0.5 * torch.randn(*[1]*len(x.shape[:-3]),
                                                 self.R,
                                                 self.input_dim,
                                                 device=self.device)
       
        y_quant, b_up, b_low = quantize_uniform(
            (self.dither(self.compress(x))).cpu().numpy(),
            n_bits=self.nbits)[1:4]

        y_quant, b_up, b_low = torch.from_numpy(
            y_quant).to(self.device), torch.from_numpy(b_up).to(self.device), \
            torch.from_numpy(b_low).to(self.device)


        return y_quant, b_up, b_low

    def error(self, x, y):
        y_quant, b_up, b_low = y
        loss = loglikelihood_quantization(self.compress(x),
                                          self.sigma, b_up, b_low)
        return loss

    def loss(self, x, y):
        return self.error(x, y).sum()
    
                
                


if __name__ == "__main__":
    x = torch.randn(12, 3, 256, 256).to('cuda')
    operator = CompressionQuantization(4, 3)
    y = operator(x)
    operator.error(x, y)
    operator.loss(x, y)
