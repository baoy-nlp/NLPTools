#!/usr/bin/env bash
cd ..
python -m core_nlp.interface.parser_main \
    --write-vocab testdata/toy.vocab.json \
    --train testdata/toy.clean