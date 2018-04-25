#! /bin/sh
cd ../..
export CUDA_VISIBLE_DEVICES=1
TRAIN_PATH=data/s2b/train.s2b
DEV_PATH=data/s2b/dev.s2b
EXP_PATH=./s2b-3lstm-bi
python2 seq2seq/main.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXP_PATH --bidirectional --rnn_layers 3
