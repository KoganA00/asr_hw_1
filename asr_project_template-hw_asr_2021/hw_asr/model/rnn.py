from torch import nn
from torch.nn import Sequential

from hw_asr.base import BaseModel


class RNNModel(BaseModel):
    def __init__(self, n_feats, n_class, hidden_size=128, num_layers=3, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)

        self.rnn = nn.RNN(input_size=n_feats, hidden_size=hidden_size, num_layers=num_layers)
        self.out = nn.Linear(in_features=hidden_size, out_features=n_class)



    def forward(self, spectrogram, *args, **kwargs):
        output, _ = self.rnn(spectrogram)
        return {"logits": self.out(output)}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here