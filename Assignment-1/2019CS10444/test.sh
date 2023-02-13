#!/bin/bash

python3 src/main.py --model_path $1 --test_file $2 --outfile $3 --model=lr --operation=test --tfidf_file=tfidf.pkl