#!/usr/bin/env bash
python -m core_nlp.interface.parser_main \
    --model ~/experiments/pytorch-origin-gpu.model \
    --gpu-id 0 \
    --vocab ~/experiments/transformer-ptb.vocab \
    --train /home/user_data/baoy/data/con/ptb/train.clean \
    --dev /home/user_data/baoy/data/con/ptb/dev.clean