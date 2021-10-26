from typing import List, Tuple

import torch
from torch import Tensor
from hw_asr.text_encoder.char_text_encoder import CharTextEncoder
import os

import youtokentome


class CTCBPETextEncoder(CharTextEncoder):

    def __init__(self):
        self.EMPTY_TOK = '_'
        model_path = 'asr_hw_1/asr_project_template-hw_asr_2021/hw_asr/bpe/bpe_train_libreespeech.model'
        self.bpe_tokenizer = youtokentome.BPE(model_path)

        alphabet = self.bpe_tokenizer.vocab()

        super().__init__(alphabet)

    def ctc_decode(self, inds):
        # TODO: your code here

        line = []
        prev_letter = None
        prev_letter_was_empty_token = False
        for ind in inds:
            if torch.is_tensor(ind):
                ind = ind.item()
            if ind == 0:
                prev_letter_was_empty_token = True
                continue
            if len(line) == 0 or prev_letter != ind or prev_letter_was_empty_token:
                line.append(ind)
                prev_letter = line[-1]
                prev_letter_was_empty_token = False

        return self.bpe_tokenizer.decode([line])[0]

    def encode(self, text):
        text = self.normalize_text(text)
        try:
            return Tensor(self.bpe_tokenizer.encode([text], output_type=youtokentome.OutputType.ID))
        except KeyError as e:
            raise Exception(
                f"Can't encode text '{text}'")
