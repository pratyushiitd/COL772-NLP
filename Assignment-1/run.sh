#!/bin/bash

python3 src/main.py \
    --train_file data/train.csv \
    --lemmatize=1 \
    --stem=0 \
    --remove_stopwords=1 \
    --test_size 0.2 \
    --random_state 42 \
    --out_file data/train_clean.csv \
    --model lr \
    --ngrams 2 \
    --min_df 5 \
    --max_df 0.95 \
    --tfidf_max_feat 50000 \

