import numpy as np
import torch
from .registry import register_operator
from .base import Operator

# Adapted from: https://github.com/zhangbingliang2019/DAPS/blob/main/forward_operator/__init__.py
# Original author: bingliang
def random_sq_bbox(img, mask_shape, image_size=256, margin=(16, 16)):
    """Generate a random sqaure mask for inpainting
    """
    B, C, H, W = img.shape
    h, w = mask_shape
    margin_height, margin_width = margin
    maxt = image_size - margin_height - h
    maxl = image_size - margin_width - w

    # bb
    t = np.random.randint(margin_height, maxt)
    l = np.random.randint(margin_width, maxl)

    # make mask
    mask = torch.ones([B, C, H, W], device=img.device)
    mask[..., t:t + h, l:l + w] = 0

    return mask, t, t + h, l, l + w


# Adapted from: https://github.com/zhangbingliang2019/DAPS/blob/main/forward_operator/__init__.py
# Original author: bingliang
class mask_generator:
    def __init__(self, mask_type, mask_len_range=None, mask_prob_range=None,
                 image_size=256, margin=(32, 32)):
        """
        (mask_len_range): given in (min, max) tuple.
        Specifies the range of box size in each dimension
        (mask_prob_range): for the case of random masking,
        specify the probability of individual pixels being masked
        """
        assert mask_type in ['box', 'random', 'both', 'extreme', 'whole']
        self.mask_type = mask_type
        self.mask_len_range = mask_len_range
        self.mask_prob_range = mask_prob_range
        self.image_size = image_size
        self.margin = margin

    def _retrieve_box(self, img):
        l, h = self.mask_len_range
        l, h = int(l), int(h)
        mask_h = np.random.randint(l, h)
        mask_w = np.random.randint(l, h)
        mask, t, tl, w, wh = random_sq_bbox(img,
                                            mask_shape=(mask_h, mask_w),
                                            image_size=self.image_size,
                                            margin=self.margin)
        return mask, t, tl, w, wh

    def _retrieve_random(self, img):
        total = self.image_size ** 2
        # random pixel sampling
        l, h = self.mask_prob_range
        prob = np.random.uniform(l, h)
        mask_vec = torch.ones([1, self.image_size * self.image_size])
        samples = np.random.choice(
            self.image_size * self.image_size, int(total * prob), replace=False)
        mask_vec[:, samples] = 0
        mask_b = mask_vec.view(1, self.image_size, self.image_size)
        mask_b = mask_b.repeat(3, 1, 1)
        mask = torch.ones_like(img, device=img.device)
        mask[:, ...] = mask_b
        return mask

    def __call__(self, img):
        if self.mask_type == 'random':
            mask = self._retrieve_random(img)
            return mask
        elif self.mask_type == 'box':
            mask, t, th, w, wl = self._retrieve_box(img)
            return mask
        elif self.mask_type == 'extreme':
            mask, t, th, w, wl = self._retrieve_box(img)
            mask = 1. - mask
            return mask
        elif self.mask_type == 'whole':
            mask = torch.zeros_like(img)
            return mask


# Adapted from: https://github.com/zhangbingliang2019/DAPS/blob/main/forward_operator/__init__.py
# Original author: bingliang
@register_operator(name='inpainting')
class Inpainting(Operator):
    def __init__(self, mask_type, mask_len_range=None, mask_prob_range=None, resolution=256, device='cuda',
                 sigma=0.05):
        super().__init__(sigma)
        self.mask_gen = mask_generator(
            mask_type, mask_len_range, mask_prob_range, resolution)
        self.mask = None  # [B, 1, H, W]

    def __call__(self, x):
        if self.mask is None:
            self.mask = self.mask_gen(x)
            self.mask = self.mask[0:1, 0:1, :, :]
        return x * self.mask
