from __future__ import print_function, division

import torch
import torchtext

from analysis_utils import write_docs, eval_f1score2
from global_names import GlobalNames
from loss import NLLLoss


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

    def evaluate2(self, model, data, k=10):
        """ Evaluate a model on given dataset and return performance.

        Args:
            model (seq2seq.models): model to evaluate
            data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against

        Returns:
            loss (float): loss of the given model on the given dataset
        """

        def extract_batch_predict(tgt_id_seq, tgt_vocab):
            process = []
            for tok in tgt_id_seq:
                if tok == sos or tok == pad:
                    pass
                elif tok == eos:
                    break
                else:
                    process.append(tgt_vocab.itos[tok])
            return " ".join(process)

        model.eval()
        result = []
        ref = []
        loss = self.loss
        loss.reset()

        device = None if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=True, sort_key=lambda x: len(x.src),
            device=device, train=False)
        tgt_vocab = data.fields[GlobalNames.tgt_field_name].vocab
        pad = tgt_vocab.stoi[data.fields[GlobalNames.tgt_field_name].pad_token]
        eos = model.decoder.eos_id
        sos = model.decoder.sos_id
        for batch in batch_iterator:
            input_variables, input_lengths = getattr(batch, GlobalNames.src_field_name)
            target_variables = getattr(batch, GlobalNames.tgt_field_name)

            decoder_outputs, decoder_hidden, other = model(input_variables, input_lengths.tolist(), target_variables)

            # Evaluation
            seqlist = other['sequence']
            for step, step_output in enumerate(decoder_outputs):
                target = target_variables[:, step + 1]
                loss.eval_batch(step_output.view(target_variables.size(0), -1), target)

            result_tensor = torch.cat(seqlist, dim=1)
            batch_size = input_lengths.size(0)
            for i in range(batch_size):
                pred = result_tensor[i].cpu() if torch.cuda.is_available() else result_tensor[i]
                pred = pred.data.numpy()
                result.append(extract_batch_predict(pred[:other['length'][i]], tgt_vocab))

                target = target_variables[i].cpu() if torch.cuda.is_available() else target_variables[i]
                target = target.data.numpy()
                ref.append(extract_batch_predict(target, tgt_vocab))
        write_docs(GlobalNames.mid_res_file, docs=result)
        write_docs(GlobalNames.mid_dev_file, docs=ref)
        accuracy = eval_f1score2(GlobalNames.mid_res_file, GlobalNames.mid_dev_file, GlobalNames.fm)

        return loss.get_loss(), accuracy

    def test(self, model, data):
        """ Evaluate a model on given dataset and return performance.

        Args:
            model (seq2seq.models): model to evaluate
            data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against

        Returns:
            loss (float): loss of the given model on the given dataset
        """

        def extract_batch_predict(tgt_id_seq, tgt_vocab):
            process = []
            for tok in tgt_id_seq:
                if tok == sos or tok == pad:
                    pass
                elif tok == eos:
                    break
                else:
                    process.append(tgt_vocab.itos[tok])
            return " ".join(process)

        model.eval()
        result = []
        ref = []

        device = None if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=True, sort_key=lambda x: len(x.src),
            device=device, train=False)
        tgt_vocab = data.fields[GlobalNames.tgt_field_name].vocab
        pad = tgt_vocab.stoi[data.fields[GlobalNames.tgt_field_name].pad_token]
        eos = model.decoder.eos_id
        sos = model.decoder.sos_id
        for batch in batch_iterator:
            input_variables, input_lengths = getattr(batch, GlobalNames.src_field_name)
            target_variables = getattr(batch, GlobalNames.tgt_field_name)

            decoder_outputs, decoder_hidden, other = model(input_variables, input_lengths.tolist())

            # Evaluation
            seqlist = other['sequence']
            result_tensor = torch.cat(seqlist, dim=1)
            batch_size = input_lengths.size(0)
            for i in range(batch_size):
                pred = result_tensor[i].cpu() if torch.cuda.is_available() else result_tensor[i]
                pred = pred.data.numpy()
                result.append(extract_batch_predict(pred[:other['length'][i]], tgt_vocab))

                target = target_variables[i].cpu() if torch.cuda.is_available() else target_variables[i]
                target = target.data.numpy()
                ref.append(extract_batch_predict(target, tgt_vocab))
        write_docs(GlobalNames.mid_res_file, docs=result)
        write_docs(GlobalNames.mid_dev_file, docs=ref)
        print(GlobalNames.mid_res_file)
        accuracy = eval_f1score2(GlobalNames.mid_res_file, GlobalNames.mid_dev_file, GlobalNames.fm)

        return accuracy
