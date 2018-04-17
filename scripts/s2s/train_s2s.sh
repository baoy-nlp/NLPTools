#! /bin/sh
# N.B.: assumes script is called from parent directory, as described in README.md
#cd scripts
#python generate_toy_data.py
#cd ../..
#TRAIN_PATH=test_data/test.txt
#DEV_PATH=test_data/test.txt
export CUDA_VISIBLE_DEVICES=3
TRAIN_PATH=/home/user_data/baoy/mt4par/train/train.data
DEV_PATH=/home/user_data/baoy/mt4par/dev/dev.data
EXP_PATH=../experiment
# Start training
python2 seq2seq/main.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXP_PATH
