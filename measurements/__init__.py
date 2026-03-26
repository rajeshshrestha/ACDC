from .registry import get_operator
from .blur import GaussianBlur, MotionBlur
from .hdr import HighDynamicRange
from .downsample import DownSampling
from .inpainting import Inpainting
from .phaseretrieval import PhaseRetrieval
from .compressionquantization import CompressionQuantization

__all__ = [get_operator, GaussianBlur, MotionBlur, HighDynamicRange,
           DownSampling, Inpainting, PhaseRetrieval, CompressionQuantization]
