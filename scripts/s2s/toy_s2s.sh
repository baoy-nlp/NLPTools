#! /bin/sh
cd ../..
export CUDA_VISIBLE_DEVICES=0
TRAIN_PATH=data/s2s/dev.s2s
DEV_PATH=data/s2s/dev.s2s
EXPT_DIR=./debug-model
python2 seq2seq/main.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_DIR