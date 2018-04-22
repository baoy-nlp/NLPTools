#! /bin/sh
cd ../..
export CUDA_VISIBLE_DEVICES=1
TRAIN_PATH=data/s2s/train.s2s
DEV_PATH=data/s2s/dev.s2s
EXP_PATH=./origin
python2 seq2seq/main.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXP_PATH
