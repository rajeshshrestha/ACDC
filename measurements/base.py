from abc import ABC, abstractmethod
import torch

# Adapted from: https://github.com/zhangbingliang2019/DAPS/blob/main/forward_operator/__init__.py
# Original author: bingliang
# Changes: Modified function add compute loss and post_ml_op

class Operator(ABC):
    def __init__(self, sigma):
        self.sigma = sigma

    @abstractmethod
    def __call__(self, x):
        pass

    def measure(self, x):
        y0 = self(x)
        return y0 + self.sigma * torch.randn_like(y0)

    def error(self, x, y):
        return ((self(x) - y) ** 2).flatten(1).sum(-1)
    
    

    def log_likelihood(self, x, y):
        return -self.error(x, y) / 2 / self.sigma ** 2

    def likelihood(self, x, y):
        return torch.exp(self.log_likelihood(x, y))
    
    def loss(self, x, y):
        return self.error(x,y).sum()/(2*self.sigma**2)
    
    def post_ml_op(self, x, y):
        return x