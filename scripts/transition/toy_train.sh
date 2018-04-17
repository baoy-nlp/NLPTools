#!/usr/bin/env bash
cd ..
python -m core_nlp.interface.parser_main \
    --model testdata/toy.model \
    --epochs 50 \
    --batch-size 2 \
    --vocab testdata/toy.vocab.json \
    --train testdata/toy.clean \
    --dev testdata/toy.clean