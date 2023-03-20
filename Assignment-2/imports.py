import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import os, pickle, json
from torchtext.legacy.data import Field, Example, BucketIterator
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from collections import Counter
import torch.nn.functional as F
from sklearn.metrics import f1_score
import re
import numpy as np
from tqdm import tqdm

WORD_EMBEDDING_DIM = 300
WORD_HIDDEN_DIM = 150
CHAR_EMBEDDING_DIM = 50
CHAR_HIDDEN_DIM = 25
USE_CHAR = False
DROPOUT = 0.5
EPOCHS = 10
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 32
INIT_LEARNING_RATE = 0.01
START_TAG = "<START>"
STOP_TAG = "<STOP>"
USE_START_STOP = False
USE_NUMBER_NORMALIZATION = True
SAVE_FILE = "model.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
def get_mask(sents):
    mask = sents.eq(0)
    mask = mask.eq(0)
    return mask
#https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
