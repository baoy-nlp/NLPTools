from __future__ import print_function, division

import torch
import torchtext

from global_names import GlobalNames
from loss import NLLLoss
from parser_utils import write_docs, eval_f1score


class Evaluator(object):
    """ Class to evaluate models with given datasets.

    Args:
        loss (seq2seq.loss, optional): loss for evaluator (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for evaluator (default: 64)
    """

    def __init__(self, loss=NLLLoss(), batch_size=64):
        self.loss = loss
        self.batch_size = batch_size

    def evaluate(self, model, data):
        """ Evaluate a model on given dataset and return performance.

        Args:
            model (seq2seq.models): model to evaluate
            data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against

        Returns:
            loss (float): loss of the given model on the given dataset
        """
        model.eval()

        loss = self.loss
        loss.reset()
        match = 0
        total = 0

        device = None if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=True, sort_key=lambda x: len(x.src),
            device=device, train=False)
        tgt_vocab = data.fields[GlobalNames.tgt_field_name].vocab
        pad = tgt_vocab.stoi[data.fields[GlobalNames.tgt_field_name].pad_token]

        for batch in batch_iterator:
            input_variables, input_lengths = getattr(batch, GlobalNames.src_field_name)
            target_variables = getattr(batch, GlobalNames.tgt_field_name)

            decoder_outputs, decoder_hidden, other = model(input_variables, input_lengths.tolist(), target_variables)

            # Evaluation
            seqlist = other['sequence']
            for step, step_output in enumerate(decoder_outputs):
                target = target_variables[:, step + 1]
                loss.eval_batch(step_output.view(target_variables.size(0), -1), target)

                non_padding = target.ne(pad)
                predict = seqlist[step].view(-1)
                correct = predict.eq(target).masked_select(non_padding).sum().data[0]
                match += correct
                total += non_padding.sum().data[0]

        if total == 0:
            accuracy = float('nan')
        else:
            accuracy = match / total

        return loss.get_loss(), accuracy

    def evaluate2(self, model, data):
        """ Evaluate a model on given dataset and return performance.

        Args:
            model (seq2seq.models): model to evaluate
            data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against

        Returns:
            loss (float): loss of the given model on the given dataset
        """

        def extract_batch_predict(tgt_id_seq, tgt_vocab):
            process = ([tgt_vocab.itos[tok] for tok in tgt_id_seq])
            result.append(" ".join(process))

        model.eval()
        result = []
        loss = self.loss
        loss.reset()
        match = 0
        total = 0

        device = None if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=True, sort_key=lambda x: len(x.src),
            device=device, train=False)
        tgt_vocab = data.fields[GlobalNames.tgt_field_name].vocab
        pad = tgt_vocab.stoi[data.fields[GlobalNames.tgt_field_name].pad_token]

        for batch in batch_iterator:
            input_variables, input_lengths = getattr(batch, GlobalNames.src_field_name)
            target_variables = getattr(batch, GlobalNames.tgt_field_name)

            decoder_outputs, decoder_hidden, other = model(input_variables, input_lengths.tolist(), target_variables)

            # Evaluation
            seqlist = other['sequence']
            for step, step_output in enumerate(decoder_outputs):
                target = target_variables[:, step + 1]
                loss.eval_batch(step_output.view(target_variables.size(0), -1), target)

                non_padding = target.ne(pad)
                predict = seqlist[step].view(-1)
                correct = predict.eq(target).masked_select(non_padding).sum().data[0]
                match += correct
                total += non_padding.sum().data[0]

            result_tensor = torch.cat(other['sequence'], dim=1)
            batch_size = decoder_hidden.size(1)
            for i in range(batch_size):
                length = other['length'][i]

                seq = result_tensor[i].cpu() if torch.cuda.is_available() else result_tensor[i]
                seq = seq.data.numpy()
                extract_batch_predict(seq[:length], tgt_vocab)
        write_docs(GlobalNames.mid_res_file, docs=result)
        # if total == 0:
        #     accuracy = float('nan')
        # else:
        #     accuracy = match / total
        accuracy = eval_f1score(GlobalNames.mid_res_file, GlobalNames.mid_dev_file)

        return loss.get_loss(), accuracy
