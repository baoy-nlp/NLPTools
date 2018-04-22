import torch
from torch.autograd import Variable

from TopKDecoder import TopKDecoder
from global_names import GlobalNames
from seq2seq import Seq2seq


class Predictor(object):
    def __init__(self, model, src_vocab, tgt_vocab):
        """
        Predictor class to evaluate for a given model.
        Args:
            model (seq2seq.models): trained model. This can be loaded from a checkpoint
                using `seq2seq.util.checkpoint.load`
            src_vocab (seq2seq.dataset.vocabulary.Vocabulary): source sequence vocabulary
            tgt_vocab (seq2seq.dataset.vocabulary.Vocabulary): target sequence vocabulary
        """
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.model.eval()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def predict(self, src_seq):
        """ Make prediction given `src_seq` as input.

        Args:
            src_seq (list): list of tokens in source language

        Returns:
            tgt_seq (list): list of tokens in target language as predicted
            by the pre-trained model
        """
        input_variable = Variable(torch.LongTensor([self.src_vocab.stoi[tok] for tok in src_seq]),
                                  volatile=True).view(1, -1)
        input_length = [len(src_seq)]
        if torch.cuda.is_available():
            input_variable = input_variable.cuda()

        decoder_outputs, decoder_hidden, other = self.model(input_variable, input_length)
        length = other['length'][0]

        tgt_id_seq = [other['sequence'][di][0].data[0] for di in range(length)]
        tgt_seq = [self.tgt_vocab.itos[tok] for tok in tgt_id_seq]
        return tgt_seq

    @staticmethod
    def evaluate(model, data, k=1):
        beam_search = Seq2seq(model.encoder, TopKDecoder(model.decoder, k))
        input_vocab = data.fields[GlobalNames.src_field_name].vocab
        output_vocab = data.fields[GlobalNames.tgt_field_name].vocab
        pred_machine = Predictor(beam_search, input_vocab, output_vocab)

        result = [" ".join(pred_machine.predict(item.src)) for item in data.examples]
        return result

        # beam_search = Seq2seq(seq2seq.encoder, TopKDecoder(seq2seq.decoder, 3))
        #
        # predictor = Predictor(beam_search, input_vocab, output_vocab)
        # inp_seq = "1 3 5 7 9"
        # print(inp_seq)
        # seq = predictor.predict(inp_seq.split())
        # print(seq)
        # assert " ".join(seq[:-1]) == inp_seq[::-1]
