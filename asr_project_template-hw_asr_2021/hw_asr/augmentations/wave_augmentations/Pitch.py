import librosa

import torch_audiomentations
from torch import Tensor
import torch
from hw_asr.augmentations.base import AugmentationBase


class Pitch(AugmentationBase):
    def __init__(self, min_=-4, max_=4, *args, **kwargs):
        self.min_ = min_
        self.max_ = max_

    def __call__(self, data: Tensor,  **kwargs):
        n_steps = (self.min_ + (self.max_ - self.min_) * torch.rand(1)).item()
        return torch.from_numpy(
                        librosa.effects.pitch_shift(
                            data.cpu().numpy().squeeze(), 16000,
                            n_steps
                            )
                        )
