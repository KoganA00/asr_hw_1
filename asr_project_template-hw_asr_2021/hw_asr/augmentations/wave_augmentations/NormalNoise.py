from torch import distributions
from torch import Tensor
from torch_audiomentations import AddColoredNoise

from hw_asr.augmentations.base import AugmentationBase


class NormalNoise(AugmentationBase):
    def __init__(self, min_f_decay=0,  *args, **kwargs):
        self.noiser = AddColoredNoise(mean_, std_)

    def __call__(self, data: Tensor):

        return data + self.noiser.sample(data.size())
