from typing import List, Tuple

import torch
from torch import Tensor
from hw_asr.text_encoder.char_text_encoder import CharTextEncoder
import os

#import youtokentome

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
        return self.bpe_tokenizer.decode([result])[0]

    def encode(self, text) -> Tensor:
        text = self.normalize_text(text)
        try:
            return Tensor(self.bpe.encode([text], output_type=youtokentome.OutputType.ID)).squeeze(0).squeeze(0)
        except KeyError as e:
            raise Exception(
                f"Can't encode text '{text}'")

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Tuple[str, float]]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos = []
        # TODO: your code here
        old_hypos_prob = {'': 1}

        ''''
        for i in range(char_length):

            new_hypos_prob = {}
            most_probable_inds = torch.argsort(probs[i], descending=True)[:beam_size]
            for new_ind in most_probable_inds:
                new_char = self.ind2char[new_ind.item()]
                for hypo in old_hypos_prob.keys():

                    # Формируем новую гипотезу - слово, без повторов букв
                    #Если предыдущая гипотеза пуста, то добавляем символ
                    if len(hypo) == 0:
                        new_hypo = hypo + new_char
                    #Если предыдущая буква - пустой символ и след буква пустой - то не добавляем
                    elif hypo[-1] == self.EMPTY_TOK and new_char == self.EMPTY_TOK:
                        new_hypo = hypo
                    #
                    elif hypo[-1] == self.EMPTY_TOK:
                        new_hypo = hypo[:-1] + new_char
                    elif hypo[-1] != new_char:
                        new_hypo = hypo + new_char
                    else:
                        new_hypo = hypo
                    print(new_hypo)
                    new_hypos_prob[new_hypo] = new_hypos_prob.get(new_hypo, 0) + old_hypos_prob[hypo] * probs[i][new_ind]
            old_hypos_prob = new_hypos_prob.copy()
        #
        #todo пробежаться и отсмотреть повторы

        hypos = [(x, old_hypos_prob[x]) for x in old_hypos_prob.keys()]
        '''
'''
        for i in range(char_length):

            new_hypos_prob = {}
            most_probable_inds = torch.argsort(probs[i], descending=True)[:beam_size]
            for new_ind in most_probable_inds:
                new_char = self.ind2char[new_ind.item()]
                for hypo in old_hypos_prob.keys():
                    new_hypo = hypo + new_char
                    new_hypos_prob[new_hypo] = new_hypos_prob.get(new_hypo, 0) + old_hypos_prob[hypo] * probs[i][
                        new_ind].item()
            old_hypos_prob = new_hypos_prob.copy()
            #
            # todo пробежаться и отсмотреть повторы
            final_hypos = {}
            for hypo in old_hypos_prob.keys():
                hypo_decoded = self._ctc_decode_string(hypo)
                final_hypos[hypo_decoded] = final_hypos.get(hypo_decoded, 0) + old_hypos_prob[hypo]

        hypos = [(x, final_hypos[x]) for x in final_hypos.keys()]

        return sorted(hypos, key=lambda x: x[1], reverse=True)'''