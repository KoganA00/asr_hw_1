import librosa

import torch_audiomentations
from torch import Tensor
import torch
from hw_asr.augmentations.base import AugmentationBase


class Stretch(AugmentationBase):
    def __init__(self, min_=0.75, max_=1.25, *args, **kwargs):
        self.min_ = min_
        self.max_ = max_

    def __call__(self, data: Tensor,  **kwargs):
        rate =  (self.min_ + (self.max_ - self.min_) * torch.rand(1)).item()
        return torch.from_numpy(
                        librosa.effects.time_stretch(
                            data,
                            rate
                            )
                        )
