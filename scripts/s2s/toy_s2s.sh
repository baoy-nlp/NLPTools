#! /bin/sh
cd ../..
TRAIN_PATH=data/dev.data
DEV_PATH=data/dev.data
python2 seq2seq/main.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir ../experiment