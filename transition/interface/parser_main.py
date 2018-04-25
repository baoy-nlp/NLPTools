"""
Command-line interface for Span-Based Constituency Parser.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from transition.models.parser.trainer import generate_vocab
from transition.models.parser.trainer import test
from transition.models.parser.trainer import train
from transition.utils.global_names import GlobalNames

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='Span-Based Constituency Parser')
    parser.add_argument(
        '--gpu-ids',
        dest='gpu_ids',
        default='-1',
        type=str,
        help="-1",
    )
    parser.add_argument(
        '--model',
        dest='model',
        help='File to save or load model.',
    )
    parser.add_argument(
        '--train',
        dest='train',
        help='Training trees. PTB (parenthetical) format.',
    )
    parser.add_argument(
        '--test',
        dest='test',
        help=(
            'Evaluation trees. PTB (parenthetical) format.'
            ' Omit for training.'
        ),
    )
    parser.add_argument(
        '--dev',
        dest='dev',
        help=(
            'Validation trees. PTB (parenthetical) format.'
            ' Required for training'
        ),
    )
    parser.add_argument(
        '--vocab',
        dest='vocab',
        help='JSON file from which to load vocabulary.',
    )
    parser.add_argument(
        '--write-vocab',
        dest='vocab_output',
        help='Destination to save vocabulary from training data.',
    )
    parser.add_argument(
        '--word-dims',
        dest='word_dims',
        type=int,
        default=50,
        help='Embedding dimesions for word forms. (DEFAULT=50)',
    )
    parser.add_argument(
        '--tag-dims',
        dest='tag_dims',
        type=int,
        default=20,
        help='Embedding dimesions for POS tags. (DEFAULT=20)',
    )
    parser.add_argument(
        '--lstm-units',
        dest='lstm_units',
        type=int,
        default=200,
        help='Number of LSTM units in each layer/direction. (DEFAULT=200)',
    )
    parser.add_argument(
        '--hidden-units',
        dest='hidden_units',
        type=int,
        default=200,
        help='Number of hidden units for each FC ReLU layer. (DEFAULT=200)',
    )
    parser.add_argument(
        '--epochs',
        dest='epochs',
        type=int,
        default=20,
        help='Number of training epochs. (DEFAULT=10)',
    )
    parser.add_argument(
        '--batch-size',
        dest='batch_size',
        type=int,
        default=10,
        help='Number of sentences per training update. (DEFAULT=10)',
    )
    parser.add_argument(
        '--droprate',
        dest='droprate',
        type=float,
        default=0.5,
        help='Dropout probability. (DEFAULT=0.5)',
    )
    parser.add_argument(
        '--unk-param',
        dest='unk_param',
        type=float,
        default=0.8375,
        help='Parameter z for random UNKing. (DEFAULT=0.8375)',
    )
    parser.add_argument(
        '--alpha',
        dest='alpha',
        type=float,
        default=1.0,
        help='Softmax distribution weighting for exploration. (DEFAULT=1.0)',
    )
    parser.add_argument(
        '--beta',
        dest='beta',
        type=float,
        default=0,
        help='Probability of using oracle action in exploration. (DEFAULT=0)',
    )
    parser.add_argument('--np-seed', type=int, dest='np_seed')

    args = parser.parse_args()

    if args.np_seed is not None:
        import numpy as np

        np.random.seed(args.np_seed)

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    GlobalNames.use_gpu = True if args.gpu_ids != '-1' is not None else False
    print(args)
    fm = generate_vocab(args)

    if args.model is None:
        print('Must specify --model or (or --write-vocab) parameter.')
        print('    (Use -h or --help flag for full option list.)')
        sys.exit()

    if args.test is not None:
        test(fm, args)
    elif args.train is not None:
        train(fm, args)
