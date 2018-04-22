import argparse
import logging
import os

import torch
import torchtext

from checkpoint import Checkpoint
from evaluator import Evaluator
from fields import SourceField, TargetField
from loss import Perplexity

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev_path', action='store', dest='dev_path',
                        help='Path to dev data')
    parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment',
                        help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
    parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                        help='The name of the checkpoint to load, usually an encoded time string')
    parser.add_argument('--log-level', dest='log_level',
                        default='info',
                        help='Logging level.')

    opt = parser.parse_args()

    LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
    logging.info(opt)

    # Prepare dataset
    src = SourceField()
    tgt = TargetField()
    max_len = 150


    def len_filter(example):
        return len(example.src) <= max_len and len(example.tgt) <= max_len * 3


    dev = torchtext.data.TabularDataset(
        path=opt.dev_path, format='tsv',
        fields=[('src', src), ('tgt', tgt)],
        filter_pred=len_filter
    )

    logging.info("loading checkpoint from {}".format(
        os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)))
    checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model
    src.vocab = checkpoint.input_vocab
    tgt.vocab = checkpoint.output_vocab

    weight = torch.ones(len(tgt.vocab))
    pad = tgt.vocab.stoi[tgt.pad_token]
    loss = Perplexity(weight, pad)
    if torch.cuda.is_available():
        loss.cuda()
    evaluator = Evaluator(loss=loss, batch_size=32)
    accuracy = evaluator.test(seq2seq, dev)
    print(accuracy)
