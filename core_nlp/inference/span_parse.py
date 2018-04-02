"""
implement a span parser with py-torch
"""
import torch.nn as nn


def trainer(data, loss_func, optimizer, model):
    pass


class SpanParserNN(nn.Module):
    def __init__(self,
                 fm,
                 args
                 ):
        super(SpanParserNN, self).__init__()
