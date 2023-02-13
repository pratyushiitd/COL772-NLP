#!/bin/bash

python3 src/main.py --train_file $1 --model_path $2 --tfidf_file=tfidf.pkl --model=lr --operation=train