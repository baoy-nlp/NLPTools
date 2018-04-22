#! /bin/sh
cd ../..
export CUDA_VISIBLE_DEVICES=0
DEV_PATH=data/s2t/dev.s2t
EXP_PATH=./s2t-lstm
python2 seq2seq/tester.py --dev_path $DEV_PATH --expt_dir $EXP_PATH --load_checkpoint ""
