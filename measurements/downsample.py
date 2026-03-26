from .registry import register_operator
from .base import Operator
from .resizer import Resizer

# Adapted from: https://github.com/zhangbingliang2019/DAPS/blob/main/forward_operator/__init__.py
# Original author: bingliang
@register_operator(name='down_sampling')
class DownSampling(Operator):
    def __init__(self, resolution=256, scale_factor=4, device='cuda', sigma=0.05):
        super().__init__(sigma)
        in_shape = [1, 3, resolution, resolution]
        self.down_sample = Resizer(in_shape, 1 / scale_factor).to(device)

    def __call__(self, x):
        return self.down_sample(x)