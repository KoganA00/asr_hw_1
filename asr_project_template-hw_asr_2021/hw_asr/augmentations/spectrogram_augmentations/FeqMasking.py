import torch
import torchaudio
from hw_asr.augmentations.base import AugmentationBase


class FreqMasking(AugmentationBase):
    def __init__(self, frequency_mask=16, *args, **kwargs):
        self.aug_ = torchaudio.transforms.FrequencyMasking(frequency_mask)

    def __call__(self, data, **kwargs):
        return self.aug_(data)