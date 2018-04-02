"""
implement a span parser with py-torch
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable

torch.manual_seed(1)


class SpanParserNN(nn.Module):
    def __init__(self,
                 fm,
                 args
                 ):
        super(SpanParserNN, self).__init__()
        self.word_emb = nn.Embedding(fm.total_words(), args.word_dims)
        self.tag_emb = nn.Embedding(fm.total_tags(), args.tag_dims)

        self.lstm1_hidden_size = args.word_dims + args.tag_dims
        self.lstm2_hidden_size = 2 * args.lstm_units
        self.lstm_units = args.lstm_units

        self.fwd_lstm1 = nn.LSTM(self.lstm1_hidden_size, args.lstm_units)
        self.back_lstm1 = nn.LSTM(self.lstm1_hidden_size, args.lstm_units)

        self.fwd_lstm2 = nn.LSTM(self.lstm2_hidden_size, args.lstm_units)
        self.back_lstm2 = nn.LSTM(self.lstm2_hidden_size, args.lstm_units)

        struct = 4
        label = 3
        single_span = 4 * args.lstm_units

        self.drop = nn.Dropout(args.droprate)
        self.struct_nn = nn.Sequential(
            nn.Linear(struct * single_span, args.hidden_units),
            nn.ReLU(),
            nn.Linear(args.hidden_units, 2)
        )

        self.label_nn = nn.Sequential(
            nn.Linear(label * single_span, args.hidden_units),
            nn.ReLU(),
            nn.Linear(args.hidden_units, fm.total_label_actions())
        )

    def evaluate_word(self, word_ids, tag_ids, test=False):
        sentence_word = self.prepare_sequence(word_ids)
        sentence_tag = self.prepare_sequence(tag_ids)
        word_embed = self.word_emb(sentence_word)
        tag_embed = self.tag_emb(sentence_tag)

        embeds = torch.cat((word_embed, tag_embed), dim=1)
        embeds = embeds.unsqueeze(1)

        hidden = self.initHidden(self.lstm_units)
        fwd1_out, hidden = self.fwd_lstm1(embeds, hidden)
        fwd1_out = fwd1_out.view(len(word_ids), self.lstm_units)

        hidden = self.initHidden(self.lstm_units)
        back1_out, hidden = self.back_lstm1(self.invert_tensor(embeds), hidden)
        back1_out = self.invert_tensor(back1_out.view(len(word_ids), self.lstm_units))

        lstm1_out = torch.cat((fwd1_out, back1_out), dim=1)

        hidden = self.initHidden(self.lstm_units)
        fwd2_inputs = lstm1_out
        if not test:
            fwd2_inputs = self.drop(fwd2_inputs)
        fwd2_inputs = fwd2_inputs.unsqueeze(1)
        fwd2_out, hidden = self.fwd_lstm2(fwd2_inputs, hidden)
        fwd2_out = fwd2_out.view(len(word_ids), self.lstm_units)

        hidden = self.initHidden(self.lstm_units)
        back2_inputs = self.invert_tensor(lstm1_out)
        if not test:
            back2_inputs = self.drop(back2_inputs)
        back2_inputs = back2_inputs.unsqueeze(1)
        back2_out, hidden = self.back_lstm2(back2_inputs, hidden)
        back2_out = self.invert_tensor(back2_out.view(len(word_ids), self.lstm_units))

        fwd_out = torch.cat((fwd1_out, fwd2_out), dim=1)
        back_out = torch.cat((back1_out, back2_out), dim=1)

        return fwd_out, back_out

    def forward(self, fwd_out, back_out, lefts, rights, eval_type='struct', test=False):
        fwd_span_out = []
        for left_index, right_index in zip(lefts, rights):
            fwd_span_out.append(fwd_out[right_index] - fwd_out[left_index - 1])
        fwd_span_vec = torch.cat(fwd_span_out, dim=0)

        back_span_out = []
        for left_index, right_index in zip(lefts, rights):
            back_span_out.append(back_out[left_index] - back_out[right_index + 1])
        back_span_vec = torch.cat(back_span_out, dim=0)

        hidden_input = torch.cat([fwd_span_vec, back_span_vec], dim=0)
        if not test:
            hidden_input = self.drop(hidden_input)

        if eval_type == 'struct':
            return self.struct_nn(hidden_input)
        else:
            return self.label_nn(hidden_input)

    def invert_tensor(self, tensor):
        idx = [i for i in range(tensor.size(0) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx))
        return tensor.index_select(0, idx)
        # rNpArr = np.flip(tensor.numpy(), 0).copy()  # Reverse of copy of numpy array of given tensor
        # rTensor = torch.from_numpy(rNpArr)
        # return rTensor

    def prepare_sequence(self, seq):
        tensor = torch.LongTensor(seq)
        return Variable(tensor)

    def initHidden(self, hidden_dim):
        result = (Variable(torch.zeros(1, 1, hidden_dim)),
                  Variable(torch.zeros(1, 1, hidden_dim)))
        return result
