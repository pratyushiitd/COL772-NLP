from imports import *
from utils import SentenceLoader

class NERDataset(Dataset):
    def __init__(self, vocab, path, mode):
        self.word_to_idx = vocab['word_to_idx']
        self.tag_to_idx = vocab['tag_to_idx']
        self.char_to_idx = vocab['char_to_idx']
        self.dataset = SentenceLoader(path, mode).data
        self.sentences = []
        self.char_lists = []
        self.labels = []
        for idx in range(len(self.dataset['text'])):
            sentence = [self.word_to_idx[x] if x in self.word_to_idx else self.word_to_idx['<UNK_W>'] for x in self.dataset['text'][idx]]
            label = [self.tag_to_idx[x] for x in self.dataset['tags'][idx]]
            char_list = []
            for word in self.dataset['text'][idx]:
                for char in word:
                    if char not in self.char_to_idx:
                        char_list.append(self.char_to_idx['<UNK_C>'])
                    else:
                        char_list.append(self.char_to_idx[char])
            if (USE_START_STOP):
                sentence = [self.word_to_idx[START_TAG]] + sentence + [self.word_to_idx[STOP_TAG]]   
            self.sentences.append(torch.tensor(sentence, dtype=torch.long))
            self.char_lists.append(torch.tensor(char_list, dtype=torch.long))
            self.labels.append(torch.tensor(label, dtype=torch.long))
        pass
    def __len__(self):
        return len(self.dataset['text'])

    def __getitem__(self, idx): 
        return {'text': self.sentences[idx], 'tags': self.labels[idx], 'chars': self.char_lists[idx]}