#!/bin/bash

python3 src/main.py \
    --train_file data/train_clean.csv \
    --lemmatize True \
    --stem True \
    --remove_stopwords True \
    --test_size 0.2 \
    --random_state 69 \
    --out_file data/train_clean.csv \
    --m nb \
    --ngrams 1

