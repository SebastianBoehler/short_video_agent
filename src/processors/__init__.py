"""Video and audio processors."""

from .matting import BackgroundRemover
from .compositor import VideoCompositor
from .captions import CaptionGenerator
from .stitcher import VideoStitcher

__all__ = [
    "BackgroundRemover",
    "VideoCompositor",
    "CaptionGenerator",
    "VideoStitcher",
]
