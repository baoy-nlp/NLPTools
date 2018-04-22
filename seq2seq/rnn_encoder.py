import math

import torch
import torch.nn as nn
from torch.autograd import Variable

from baseRNN import BaseRNN


class EncoderRNN(BaseRNN):
    r"""
    Applies a multi-layer RNN to an input sequence.

    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        n_layers (int, optional): number of recurrent layers (default: 1)
        bidirectional (bool, optional): if True, becomes a bidirectional encodr (defulat False)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        variable_lengths (bool, optional): if use variable length RNN (default: False)

    Inputs: inputs, input_lengths
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is a list of token IDs.
        - **input_lengths** (list of int, optional): list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)
            
    Outputs: output, hidden
        - **output** (batch, seq_len, hidden_size): tensor containing the encoded features of the input sequence
        - **hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the features in the hidden state `h`

    Examples::

         >>> encoder = EncoderRNN(input_vocab, max_seq_length, hidden_size)
         >>> output, hidden = encoder(input)

    """

    def __init__(self,
                 vocab_size,
                 max_len,
                 input_size,
                 hidden_size,
                 input_dropout_p=0,
                 dropout_p=0,
                 n_layers=3,
                 bidirectional=False,
                 rnn_cell='gru',
                 variable_lengths=False,
                 add_position_embedding=True
                 ):
        super(EncoderRNN, self).__init__(
            vocab_size, max_len, input_size, hidden_size,
            input_dropout_p, dropout_p, n_layers, rnn_cell
        )
        # self.add_position_embedding = add_position_embedding
        self.variable_lengths = variable_lengths
        self.embedding = nn.Embedding(vocab_size, input_size)
        self.rnn = self.rnn_cell(
            input_size, hidden_size, n_layers,
            batch_first=True, bidirectional=bidirectional, dropout=dropout_p
        )

    def _add_pos_embedding(self, x, min_timescale=1.0, max_timescale=1.0e4):

        batch, length, channels = list(x.size())
        assert (channels % 2 == 0)
        num_timescales = channels // 2
        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (float(num_timescales) - 1.))
        position = torch.arange(0, length).float()
        inv_timescales = torch.arange(0, num_timescales).float()
        if x.is_cuda:
            position = position.cuda()
            inv_timescales = inv_timescales.cuda()

        inv_timescales.mul_(-log_timescale_increment).exp_().mul_(min_timescale)
        scaled_time = position.unsqueeze(1).expand(
            length, num_timescales) * inv_timescales.unsqueeze(0).expand(length, num_timescales)
        # scaled time is now length x num_timescales
        # length x channels
        signal = torch.cat([scaled_time.sin(), scaled_time.cos()], 1)

        return Variable(signal.unsqueeze(0).expand(batch, length, channels), requires_grad=False)

    def forward(self, input_var, input_lengths=None):
        """
        Applies a multi-layer RNN to an input sequence.

        Args:
            input_var (batch, seq_len): tensor containing the features of the input sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch

        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
        """
        embedded = self.embedding(input_var)
        # if self.add_position_embedding:
        #     embedded += self._add_pos_embedding(embedded)
        embedded = self.input_dropout(embedded)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        output, hidden = self.rnn(embedded)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden
