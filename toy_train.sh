#!/usr/bin/env bash
python -m core_nlp.interface.parser_main \
    --model testdata/toy.model \
    --gpu-id 0 \
    --epochs 50 \
    --batch-size 2 \
    --vocab testdata/toy.vocab.json \
    --train testdata/toy.clean \
    --dev testdata/toy.clean