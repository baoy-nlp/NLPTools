"""
implement a span parser with py-torch
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from transition.inference.parser import Parser
from transition.models.parser.encoder import BILSTMEncoder
from transition.utils.global_names import GlobalNames

torch.manual_seed(1)


class Network(nn.Module):
    def __init__(self,
                 fm,
                 args,
                 ):
        super(Network, self).__init__()
        self.word_emb = nn.Embedding(fm.total_words(), args.word_dims)
        self.tag_emb = nn.Embedding(fm.total_tags(), args.tag_dims)

        single_span = 4 * args.lstm_units

        self.encoder = BILSTMEncoder(
            input_dim=args.word_dims + args.tag_dims,
            hidden_dim=args.lstm_units,
            output_dim=args.lstm_units,
            droprate=args.droprate
        )

        self.drop = nn.Dropout(args.droprate)

        self.struct_nn = nn.Sequential(
            nn.Linear(4 * single_span, args.hidden_units),
            nn.ReLU(),
            nn.Linear(args.hidden_units, 2)
        )

        self.label_nn = nn.Sequential(
            nn.Linear(3 * single_span, args.hidden_units),
            nn.ReLU(),
            nn.Linear(args.hidden_units, fm.total_label_actions())
        )
        if GlobalNames.use_gpu:
            self.cuda()

    def init_weight(self):
        """
        Init the all the parameter of model, including:
            Embedding
            Linear W
            Linear B
        """
        pass

    def prepare_sequence(self, seq):
        tensor = torch.LongTensor(seq)
        if GlobalNames.use_gpu:
            return Variable(tensor).cuda()
        else:
            return Variable(tensor)

    def evaluate_action(self, fwd_out, back_out, lefts, rights, eval_type='struct', test=False):
        fwd_span_out = []
        for left_index, right_index in zip(lefts, rights):
            fwd_span_out.append(fwd_out[right_index] - fwd_out[left_index - 1])
        fwd_span_vec = torch.cat(fwd_span_out, dim=-1)

        back_span_out = []
        for left_index, right_index in zip(lefts, rights):
            back_span_out.append(back_out[left_index] - back_out[right_index + 1])
        back_span_vec = torch.cat(back_span_out, dim=-1)

        hidden_input = torch.cat([fwd_span_vec, back_span_vec], dim=-1)
        if not test:
            hidden_input = self.drop(hidden_input)

        if eval_type == 'struct':
            return self.struct_nn(hidden_input)[0]
        else:
            return self.label_nn(hidden_input)[0]

    def lookup(self, word_ids, tag_ids):
        sentence_w = self.prepare_sequence(word_ids)
        sentence_t = self.prepare_sequence(tag_ids)
        word_embed = self.word_emb(sentence_w)
        tag_embed = self.tag_emb(sentence_t)
        embed = torch.cat([word_embed, tag_embed], dim=-1)
        return embed

    def encode(self, word_ids, tag_ids, test=False):
        inputs = self.lookup(word_ids, tag_ids)
        return self.encoder(inputs, test)

    def _explore(self, data, fm, alpha=1.0, beta=0):
        struct_data = {}
        label_data = {}

        tree = data['tree']
        sentence = tree.sentence
        w = data['w']
        t = data['t']

        n = len(sentence)
        state = Parser(n)

        fwd, back = self.encode(w, t, test=True)

        for step in range(2 * n - 1):
            features = state.s_features()
            if not state.can_combine():
                action = 'sh'
                correct_action = 'sh'
            elif not state.can_shift():
                action = 'comb'
                correct_action = 'comb'
            else:

                correct_action = state.s_oracle(tree)

                r = np.random.random()
                if r < beta:
                    action = correct_action
                else:
                    left, right = features
                    scores = self.evaluate_action(
                        fwd,
                        back,
                        left,
                        right,
                        'struct',
                        test=True,
                    )
                    if GlobalNames.use_gpu:
                        scores = scores.cpu().data.numpy()
                    else:
                        scores = scores.data.numpy()
                    # sample from distribution
                    exp = np.exp(scores * alpha)
                    softmax = exp / (exp.sum())
                    r = np.random.random()
                    if r <= softmax[0]:
                        action = 'sh'
                    else:
                        action = 'comb'
            struct_data[features] = fm.s_action_index(correct_action)
            state.take_action(action)

            features = state.l_features()
            correct_action = state.l_oracle(tree)
            label_data[features] = fm.l_action_index(correct_action)

            r = np.random.random()
            if r < beta:
                action = correct_action
            else:
                left, right = features
                scores = self.evaluate_action(
                    fwd,
                    back,
                    left,
                    right,
                    'label',
                    test=True,
                )
                if GlobalNames.use_gpu:
                    scores = scores.cpu().data.numpy()
                else:
                    scores = scores.data.numpy()
                if step < (2 * n - 2):
                    action_index = np.argmax(scores)
                else:
                    action_index = 1 + np.argmax(scores[1:])
                action = fm.l_action(action_index)
            state.take_action(action)

        predicted = state.tree()
        predicted.propagate_sentence(sentence)

        example = {
            'w': w,
            't': t,
            'struct_data': struct_data,
            'label_data': label_data,
            'accuracy': predicted.compare(tree)
        }

        return example

    def _inference(self, sentence, fm):
        n = len(sentence)
        word_ids, tag_ids = fm.index_sentences(sentence)

        fwd, back = self.encode(word_ids, tag_ids, test=True)

        state = Parser(n)

        for step in range(2 * n - 1):
            if not state.can_combine():
                action = "sh"
            elif not state.can_shift():
                action = "comb"
            else:
                left, right = state.s_features()
                scores = self.evaluate_action(
                    fwd,
                    back,
                    left,
                    right,
                    'struct',
                    test=True,
                )
                if GlobalNames.use_gpu:
                    scores = scores.cpu().data.numpy()
                else:
                    scores = scores.data.numpy()
                action_index = np.argmax(scores)
                action = fm.s_action(action_index)
            state.take_action(action)

            left, right = state.l_features()
            scores = self.evaluate_action(
                fwd,
                back,
                left,
                right,
                'label',
                test=True,
            )
            if GlobalNames.use_gpu:
                scores = scores.cpu().data.numpy()
            else:
                scores = scores.data.numpy()
            if step < (2 * n - 2):
                action_index = np.argmax(scores)
            else:
                action_index = 1 + np.argmax(scores[1:])
            action = fm.l_action(action_index)
            state.take_action(action)
        tree = state.stack[0][2][0]
        tree.propagate_sentence(sentence)
        return tree

    def forward(self, data, fm, alpha=1.0, beta=0, unk_param=0.75, test=False):
        """
        Args:
            data: tree, w, i
        Return:
            [prediction,loss]
        """
        if test:
            sentence = data['tree'].sentence
            return self._inference(sentence, fm)
        else:
            this_loss = None
            example = self._explore(data, fm, alpha, beta)
            for (i, w) in enumerate(example['w']):
                if w <= 2:
                    continue

                freq = fm.word_freq_list[w]
                drop_prob = unk_param / (unk_param + freq)
                r = np.random.random()
                if r < drop_prob:
                    example['w'][i] = 0

            fwd, back = self.encode(example['w'], example['t'])
            for (left, right), correct in example['struct_data'].items():
                scores = self.evaluate_action(fwd, back, left, right, 'struct')
                probs = F.softmax(scores, dim=0)
                loss = -torch.log(probs[correct])
                if this_loss is not None:
                    this_loss += loss
                else:
                    this_loss = loss

            for (left, right), correct in example['label_data'].items():
                scores = self.evaluate_action(fwd, back, left, right, 'label')
                probs = F.softmax(scores, dim=0)
                loss = -torch.log(probs[correct])
                this_loss += loss
            return this_loss, example
