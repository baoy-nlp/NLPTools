#!/usr/bin/env bash
cd ../..
python -m transition.interface.parser_main \
    --model trans/toy.model \
    --gpu-id 0 \
    --vocab trans/toy.vocab.json \
    --test trans/toy.clean