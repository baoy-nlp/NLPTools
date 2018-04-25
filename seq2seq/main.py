import argparse
import logging
import os

import torch
import torchtext

from checkpoint import Checkpoint
from fields import SourceField, TargetField
from global_names import GlobalNames
from loss import CrossEntropyLoss
from optim import Optimizer
from predictor import Predictor
from rnn_decoder import DecoderRNN
from rnn_encoder import EncoderRNN
from seq2seq import Seq2seq
from trainer import SupervisedTrainer


def len_filter(example):
    return len(example.src) <= max_len and len(example.tgt) <= max_len * 3


if __name__ == "__main__":
    try:
        raw_input  # Python 2
    except NameError:
        raw_input = input  # Python 3

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_path', action='store', dest='train_path',
                        help='Path to train data')
    parser.add_argument('--dev_path', action='store', dest='dev_path',
                        help='Path to dev data')
    parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment',
                        help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
    parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                        help='The name of the checkpoint to load, usually an encoded time string')
    parser.add_argument('--resume', action='store_true', dest='resume', default=False,
                        help='Indicates if training has to be resumed from the latest checkpoint')
    parser.add_argument('--log-level', dest='log_level', default='info', help='Logging level.')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=50)
    parser.add_argument('--embed_dim', dest='embed_dim', type=int, default=512)
    parser.add_argument('--hidden_dim', dest='hidden_dim', type=int, default=256)
    parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=20000)
    parser.add_argument('--bidirectional', dest='bidirectional', action='store_true')
    parser.add_argument('--teacher_forcing_ration', dest='teacher_forcing_ratio', type=int, default=1.0)
    parser.add_argument('--max_len', dest='max_len', type=int, default=150)
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3)
    parser.add_argument('--grad_norm', dest='grad_norm', type=float, default=5.0)
    parser.add_argument('--rnn_layers', dest='rnn_layers', type=int, default=1)

    opt = parser.parse_args()
    batch_size = opt.batch_size
    input_size = opt.embed_dim
    hidden_size = opt.hidden_dim
    number_epochs = opt.num_epochs
    bidirectional = opt.bidirectional
    teacher_forcing_ratio = opt.teacher_forcing_ratio
    max_len = opt.max_len

    LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
    logging.info(opt)

    src = SourceField()
    tgt = TargetField()

    train = torchtext.data.TabularDataset(
        path=opt.train_path, format='tsv',
        fields=[(GlobalNames.src_field_name, src), (GlobalNames.tgt_field_name, tgt)],
        filter_pred=len_filter
    )
    dev = torchtext.data.TabularDataset(
        path=opt.dev_path, format='tsv',
        fields=[(GlobalNames.src_field_name, src), (GlobalNames.tgt_field_name, tgt)],
        filter_pred=len_filter
    )

    optimizer = None

    if opt.load_checkpoint is not None:
        logging.info("loading checkpoint from {}".format(
            os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)))
        checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
        checkpoint = Checkpoint.load(checkpoint_path)
        seq2seq = checkpoint.model
        input_vocab = checkpoint.input_vocab
        output_vocab = checkpoint.output_vocab
        src.vocab = input_vocab
        tgt.vocab = output_vocab
    else:

        src.build_vocab(train, max_size=50000)
        tgt.build_vocab(train, max_size=300)

        input_vocab = src.vocab
        output_vocab = tgt.vocab

        seq2seq = None

        if not opt.resume:
            encoder = EncoderRNN(
                vocab_size=len(src.vocab),
                max_len=max_len,
                input_size=input_size,
                hidden_size=hidden_size,
                n_layers=opt.rnn_layers,
                rnn_cell='lstm',
                bidirectional=bidirectional,
                variable_lengths=True,
                add_position_embedding=False,
            )
            decoder = DecoderRNN(
                vocab_size=len(tgt.vocab),
                max_len=max_len * 3,
                input_size=input_size,
                hidden_size=hidden_size * 2 if bidirectional else hidden_size,
                n_layers=opt.rnn_layers,
                rnn_cell='lstm',
                dropout_p=0.2,
                use_attention=True,
                bidirectional=bidirectional,
                eos_id=tgt.eos_id,
                sos_id=tgt.sos_id
            )
            seq2seq = Seq2seq(encoder, decoder)
            if torch.cuda.is_available():
                seq2seq.cuda()

            for param in seq2seq.parameters():
                param.data.uniform_(-0.08, 0.08)

    optimizer = Optimizer(
        torch.optim.Adam(seq2seq.parameters(), lr=opt.lr, betas=(0.9, 0.995)),
        max_grad_norm=opt.grad_norm)
    # scheduler = StepLR(optimizer.optimizer, 1)
    # optimizer.set_scheduler(scheduler)

    weight = torch.ones(len(tgt.vocab))
    pad = tgt.vocab.stoi[tgt.pad_token]
    loss = CrossEntropyLoss(weight, pad)
    if torch.cuda.is_available():
        loss.cuda()

    t = SupervisedTrainer(
        loss=loss,
        batch_size=batch_size,
        checkpoint_every=100,
        print_every=10,
        expt_dir=opt.expt_dir
    )

    seq2seq = t.train(
        seq2seq, train,
        num_epochs=number_epochs,
        dev_data=dev,
        optimizer=optimizer,
        teacher_forcing_ratio=teacher_forcing_ratio,
        resume=opt.resume
    )

    predictor = Predictor(seq2seq, input_vocab, output_vocab)

    while True:
        seq_str = raw_input("Type in a source sequence:")
        seq = seq_str.strip().split()
        print(predictor.predict(seq))
