import torch_audiomentations
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class WhiteNoise(AugmentationBase):
    def __init__(self, sr, min_f_decay=0, *args, **kwargs):
        self._aug = torch_audiomentations.AddColoredNoise(min_f_decay=min_f_decay, sample_rate=sr, *args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
