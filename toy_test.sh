#!/usr/bin/env bash
python -m core_nlp.interface.parser_main \
    --model testdata/toy.model \
    --gpu-id 0 \
    --vocab testdata/toy.vocab.json \
    --test testdata/toy.clean