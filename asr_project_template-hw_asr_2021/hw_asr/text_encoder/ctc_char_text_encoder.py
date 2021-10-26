from typing import List, Tuple

import torch

from hw_asr.text_encoder.char_text_encoder import CharTextEncoder


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str]):
        super().__init__(alphabet)
        self.ind2char = {
            0: self.EMPTY_TOK
        }
        for text in alphabet:
            self.ind2char[max(self.ind2char.keys()) + 1] = text
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        # TODO: your code here
        line = ''
        prev_letter = None
        prev_letter_was_empty_token = False
        for ind in inds:
            if torch.is_tensor(ind):
                ind = ind.item()
            if ind == 0:
                prev_letter_was_empty_token = True
                continue
            if len(line) == 0 or prev_letter != self.ind2char[ind] or prev_letter_was_empty_token:
                line = line + self.ind2char[ind]
                prev_letter = line[-1]
                prev_letter_was_empty_token = False
        return line

    def _ctc_decode_string(self, str):
        str_with_empty = []
        for char in str:
            if len(str_with_empty) > 0 and str_with_empty[-1] == char:
                continue
            str_with_empty.append(char)
        return ''.join(list(filter(lambda x: x != self.EMPTY_TOK, str_with_empty)))

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
        #print(char_length)
        for i in range(char_length):
            # print(i, end=' ')
            new_hypos_prob = {}

            for new_ind in range(voc_size):
                new_char = self.ind2char[new_ind]
                # len_ = len(old_hypos_prob.keys())
                for jj, hypo in enumerate(old_hypos_prob.keys()):
                    # print(i, '/', char_length, new_ind, '/', voc_size, jj, '/', len_)
                    # Если это первая итерация, то новый символ - это и есть новая гипотеза
                    if len(hypo) == 0:
                        new_hypo = new_char
                    # Если новый символ равен предыдущему, то гипотеза не меняется
                    elif new_char == hypo[-1]:
                        new_hypo = hypo
                    # Если предыдущий символ не пустой, то новый символ (будь то пустой символ или другая буква)
                    # просто добавляется
                    elif hypo[-1] != self.EMPTY_TOK:
                        new_hypo = hypo + new_char
                    # Если последний символ пустой, а добавляется непустой, то пустой удаляется и добавляется новый
                    elif hypo[-1] == self.EMPTY_TOK and new_char != self.EMPTY_TOK:
                        new_hypo = hypo[:-1] + new_char
                    else:
                        raise ValueError(
                            'Strange Pokemon. I checked all possible cases. Hypo: \"' + hypo + '\" New char: \'' + new_char + '\'')
                    new_hypos_prob[new_hypo] = new_hypos_prob.get(new_hypo, 0) + old_hypos_prob[hypo] * probs[i][
                        new_ind].item()
            best_hypos_beam = [k for k, v in sorted(new_hypos_prob.items(), key=lambda x: x[1])][-beam_size:]
            old_hypos_prob = {k: new_hypos_prob[k] for k in best_hypos_beam}
            # assert len(old_hypos_prob) == beam_size
            #
            # todo пробежаться и отсмотреть повторы
        final_hypos = {}
        for hypo in old_hypos_prob.keys():
            if len(hypo) > 0 and hypo[-1] == self.EMPTY_TOK:
                final_hypos[hypo[:-1]] = final_hypos.get(hypo, 0) + old_hypos_prob[hypo]
            else:
                final_hypos[hypo] = final_hypos.get(hypo, 0) + old_hypos_prob[hypo]

        hypos = [(x, final_hypos[x]) for x in final_hypos.keys()]

        return sorted(hypos, key=lambda x: x[1], reverse=True)