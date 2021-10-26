from typing import List, Tuple

import torch
from torch import Tensor
from hw_asr.text_encoder.char_text_encoder import CharTextEncoder
import os

import youtokentome

class CTCBPETextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"
    def __init__(self, alphabet: List[str], **kwargs):
        model_path = './bpe/bpe_train_libreespeech.model'
        self.bpe_tokenizer = youtokentome.BPE(model_path)

        alphabet = self.bpe_tokenizer.vocab()
        alphabet[0] = self.EMPTY_TOK
        super().__init__(alphabet)


    def ctc_decode(self, inds: List[int]) -> str:
        # TODO: your code here
        if torch.is_tensor(inds):
            inds = inds.tolist()
        line = []
        prev_letter = None
        prev_letter_was_empty_token = False
        for ind in inds:
            if ind == 0:
                prev_letter_was_empty_token = True
                continue
            if len(line) == 0 or prev_letter != ind or prev_letter_was_empty_token:
                line.append(ind)
                prev_letter = line[-1]
                prev_letter_was_empty_token = False
        return self.bpe_tokenizer.decode([line])[0]

    def encode(self, text) -> Tensor:
        text = self.normalize_text(text)
        try:
            return Tensor(self.bpe_tokenizer.encode([text], output_type=youtokentome.OutputType.ID)).squeeze(0).squeeze(0)
        except KeyError as e:
            raise Exception(
                f"Can't encode text '{text}'")
