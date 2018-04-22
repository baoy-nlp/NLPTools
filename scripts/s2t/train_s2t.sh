#! /bin/sh
cd ../..
export CUDA_VISIBLE_DEVICES=0
TRAIN_PATH=data/s2t/train.s2t
DEV_PATH=data/s2t/dev.s2t
EXP_PATH=./origin-s2t
python2 seq2seq/main.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXP_PATH --bidirectional
