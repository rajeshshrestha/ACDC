from .registry import register_operator
from .base import Operator
import torch
import torch.nn.functional as F
from .fastmri_utils import fft2c_new
import cv2
import numpy as np
from copy import deepcopy


# Adapted from: https://github.com/zhangbingliang2019/DAPS/blob/main/forward_operator/__init__.py
# Original author: bingliang
@register_operator(name='phase_retrieval')
class PhaseRetrieval(Operator):
    def __init__(self, oversample=0.0, resolution=256, sigma=0.05):
        super().__init__(sigma)
        self.pad = int((oversample / 8.0) * resolution)

    def __call__(self, x):
        x = x * 0.5 + 0.5  # [-1, 1] -> [0, 1]
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad))
        if not torch.is_complex(x):
            x = x.type(torch.complex64)
        fft2_m = torch.view_as_complex(fft2c_new(torch.view_as_real(x)))
        amplitude = fft2_m.abs()
        return amplitude

    def get_more_aligned_option(self, template, options):
        template = deepcopy(template)
        options = deepcopy(options)

        template = ((template.detach().cpu().numpy()+1) /
                    2 * 255).astype(np.uint8)

        for idx in range(len(options)):
            options[idx] = ((options[idx].detach().cpu().numpy()+1)/2
                            * 255).astype(np.uint8)
        
        ''' Use pixel similarity'''
        result_lst = []
        for option in options:
            diff = ((template - option)**2).sum()
            result_lst.append(diff)
            
        min_idx = result_lst.index(min(result_lst))
        return min_idx