import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataset import NERDataset
import sys, pickle

model, vocab = None, None
with open('model.pkl', 'rb') as f:
        state = pickle.load(f)

val_path = 'data/dev.txt'
model = state['model']
vocab = state['vocab']

val_dataset = NERDataset(vocab, val_path, "Val")
val_iterator = DataLoader(val_dataset, batch_size=32)

for batch in val_iterator:
        print(batch)
        break



