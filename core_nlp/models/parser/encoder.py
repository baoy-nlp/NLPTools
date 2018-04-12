"""
implement a span parser with py-torch
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable

from core_nlp.utils.global_names import GlobalNames


class BiLstmBaseEncoder(nn.Module):
    def __init__(self, input_dims, lstm_dims):
        super(BiLstmBaseEncoder, self).__init__()
        self.fwd_lstm = nn.LSTM(input_dims, lstm_dims)
        self.back_lstm = nn.LSTM(input_dims, lstm_dims)
        self.lstm_dims = lstm_dims

    def invert_tensor(self, tensor):
        idx = [i for i in range(tensor.size(0) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx))
        if GlobalNames.use_gpu:
            idx = idx.cuda()
        return tensor.index_select(0, idx)

    def init_hidden(self):
        var = (Variable(torch.zeros(1, 1, self.lstm_dims)), Variable(torch.zeros(1, 1, self.lstm_dims)))
        if GlobalNames.use_gpu:
            var = (var[0].cuda(), var[1].cuda())
        return var

    def forward(self, inputs):
        inputs = inputs.unsqueeze(1)  # compact the tensor to
        hx = self.init_hidden()
        fwd_out, hidden = self.fwd_lstm(inputs, hx)
        back_x = self.invert_tensor(inputs)
        back_out, hidden = self.back_lstm(back_x, hx)
        back_out = self.invert_tensor(back_out)

        return fwd_out, back_out


class BILSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, droprate):
        super(BILSTMEncoder, self).__init__()
        self.drop = nn.Dropout(droprate)
        self.word_lstm = BiLstmBaseEncoder(input_dim, hidden_dim)
        self.syn_lstm = BiLstmBaseEncoder(2 * hidden_dim, output_dim)

    def forward(self, inputs, test=False):
        slen = inputs.shape[0]
        input1 = inputs
        fwd1, back1 = self.word_lstm(input1)
        input2 = torch.cat([fwd1.view(slen, -1), back1.view(slen, -1)], dim=-1)
        if not test:
            input2 = self.drop(input2)
        fwd2, back2 = self.syn_lstm(input2)

        fwd = torch.cat([fwd1, fwd2], dim=-1)
        back = torch.cat([back1, back2], dim=-1)

        return fwd, back
