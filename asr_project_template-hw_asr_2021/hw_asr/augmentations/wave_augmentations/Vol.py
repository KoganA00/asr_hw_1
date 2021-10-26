import torch
from torchaudio.transforms import Vol
from hw_asr.augmentations.base import AugmentationBase

class Volume(AugmentationBase):
    def __init__(self, min_=0.5, max_=3, *args, **kwargs):
        self.min_ = min_
        self.max_ = max_

    def __call__(self, data):
        gain = (self.min_ + (self.max_ - self.min_) * torch.rand(1)).item()
        return Vol(gain=gain)(data.unsqueeze(1)).squeeze(1)
