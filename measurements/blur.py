from .registry import register_operator
from .base import Operator
import torch
import torch.nn as nn
import numpy as np
import scipy
from .motionblur.motionblur import Kernel
import yaml



# Adapted from: https://github.com/zhangbingliang2019/DAPS/blob/main/forward_operator/__init__.py
# Original author: bingliang
class Blurkernel(nn.Module):
    def __init__(self, blur_type='gaussian', kernel_size=31, std=3.0, device=None):
        super().__init__()
        self.blur_type = blur_type
        self.kernel_size = kernel_size
        self.std = std
        self.device = device
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(self.kernel_size // 2),
            nn.Conv2d(3, 3, self.kernel_size, stride=1,
                      padding=0, bias=False, groups=3)
        )

        self.weights_init()

    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        if self.blur_type == "gaussian":
            n = np.zeros((self.kernel_size, self.kernel_size))
            n[self.kernel_size // 2, self.kernel_size // 2] = 1
            k = scipy.ndimage.gaussian_filter(n, sigma=self.std)
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)
        elif self.blur_type == "motion":
            k = Kernel(size=(self.kernel_size, self.kernel_size),
                       intensity=self.std).kernelMatrix
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)

    def update_weights(self, k):
        if not torch.is_tensor(k):
            k = torch.from_numpy(k).to(self.device)
        for name, f in self.named_parameters():
            f.data.copy_(k)

    def get_kernel(self):
        return self.k


# Adapted from: https://github.com/zhangbingliang2019/DAPS/blob/main/forward_operator/__init__.py
# Original author: bingliang
@register_operator(name='gaussian_blur')
class GaussianBlur(Operator):
    def __init__(self, kernel_size, intensity, device='cuda', sigma=0.05):
        super().__init__(sigma)
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='gaussian',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))
        self.conv.requires_grad_(False)

    def __call__(self, data):
        return self.conv(data)


# Adapted from: https://github.com/zhangbingliang2019/DAPS/blob/main/forward_operator/__init__.py
# Original author: bingliang
@register_operator(name='motion_blur')
class MotionBlur(Operator):
    def __init__(self, kernel_size, intensity, device='cuda', sigma=0.05):
        super().__init__(sigma)
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='motion',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)  

        self.kernel = Kernel(
            size=(kernel_size, kernel_size), intensity=intensity)
        kernel = torch.tensor(self.kernel.kernelMatrix, dtype=torch.float32)
        self.conv.update_weights(kernel)
        self.conv.requires_grad_(False)

    def __call__(self, data):
        # A^T * A
        return self.conv(data)


# Adapted from: https://github.com/zhangbingliang2019/DAPS/blob/main/forward_operator/__init__.py
# Original author: bingliang
@register_operator(name='nonlinear_blur')
class NonlinearBlur(Operator):
    def __init__(self, opt_yml_path, device='cuda', sigma=0.05):
        super().__init__(sigma)
        self.device = device
        self.blur_model = self.prepare_nonlinear_blur_model(opt_yml_path)
        self.blur_model.requires_grad_(False)

        np.random.seed(0)
        kernel_np = np.random.randn(1, 512, 2, 2) * 1.2
        random_kernel = (torch.from_numpy(kernel_np)).float().to(self.device)
        self.random_kernel = random_kernel

    def prepare_nonlinear_blur_model(self, opt_yml_path):
        from .bkse.models.kernel_encoding.kernel_wizard import KernelWizard

        with open(opt_yml_path, "r") as f:
            opt = yaml.safe_load(f)["KernelWizard"]
            model_path = opt["pretrained"]
        blur_model = KernelWizard(opt)
        blur_model.eval()
        blur_model.load_state_dict(torch.load(model_path))
        blur_model = blur_model.to(self.device)
        self.random_kernel = torch.randn(1, 512, 2, 2).to(self.device) * 1.2
        return blur_model

    def call_old(self, data):
        data = (data + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        blurred = []
        for i in range(data.shape[0]):
            single_blurred = self.blur_model.adaptKernel(
                data[i:i + 1], kernel=self.random_kernel)
            blurred.append(single_blurred)
        blurred = torch.cat(blurred, dim=0)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1)  # [0, 1] -> [-1, 1]
        return blurred

    def __call__(self, data):
        data = (data + 1.0) / 2.0  # [-1, 1] -> [0, 1]

        random_kernel = self.random_kernel.repeat(data.shape[0], 1, 1, 1)
        blurred = self.blur_model.adaptKernel(data, kernel=random_kernel)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1)  # [0, 1] -> [-1, 1]
        return blurred
