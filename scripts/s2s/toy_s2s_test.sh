#! /bin/sh
export CUDA_VISIBLE_DEVICES=3
cd ../..
TRAIN_PATH=data/toy_reverse/train/data.txt
DEV_PATH=data/toy_reverse/dev/data.txt
EXPT_DIR=./test-model
python2 seq2seq/tester.py --train_path ../data/toy_reverse/train/data.txt --dev_path ../data/toy_reverse/dev/data.txt --expt_dir ../test-model