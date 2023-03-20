from imports import *

class SentenceLoader():
    def __init__(self, path, mode=None):
        self.data = self.build_sentences(path)
        print("{} Data Size: {}".format(mode, len(self.data['text'])))
    def build_sentences(self, path):
        with open(path, 'r') as f:
            sentences = []
            sentence_labels = []
            current_sentence = []
            current_labels = []
            for line in f:
                line = line.split('\t')
                if (line[0] == '\n'):
                    sentences.append(current_sentence)
                    sentence_labels.append(current_labels)
                    current_sentence = []
                    current_labels = []
                else:
                    if (USE_NUMBER_NORMALIZATION):
                        line[0] = SentenceLoader.number_normalization(line[0])
                    current_sentence += [line[0]]
                    current_labels += [line[1][:-1]]
            if (current_sentence != []):
                sentences.append(current_sentence)
                sentence_labels.append(current_labels)
            return {"text": sentences, "tags": sentence_labels}
    def number_normalization(word):
        return re.sub(r'\d', '0', word)
class WordVocabulary():
    def __init__(self, path):
        self.vocab = self.build_word_vocab(path)
        print("Word Vocab Size: ", len(self.vocab))

    def build_word_vocab(self, path):
        with open(path, 'r') as f:
            vocab = {'<PAD_W>': 0, '<UNK_W>': 1}
            if USE_START_STOP:
                vocab.update({START_TAG: 2, STOP_TAG: 3})
            for line in f:
                line = line.split('\t')
                if (line[0] == '\n'):
                    continue
                if (USE_NUMBER_NORMALIZATION):
                    line[0] = re.sub(r'\d', '0', line[0])
                if (line[0] not in vocab):
                    vocab.update({line[0]: len(vocab)})
            return vocab

class TagVocabulary():
    def __init__(self, path):
        self.vocab = self.build_tag_vocab(path)
        print("Tag Vocab Size: ", len(self.vocab))

    def build_tag_vocab(self, path):
        with open(path, 'r') as f:
            labels = {'<PAD_T>': 0}
            if USE_START_STOP:
                labels.update({START_TAG: 0, STOP_TAG: 1})
            for line in f:
                line = line.split('\t')
                if (line[0] == '\n'):
                    continue
                if (line[1][:-1] not in labels):
                    labels.update({line[1][:-1]: len(labels)})
            return labels

class CharVocabulary():
    def __init__(self, path):
        self.vocab = self.build_char_vocab(path)
        print("Char Vocab Size: ", len(self.vocab))

    def build_char_vocab(self, path):
        with open(path, 'r') as f:
            vocab = {'<PAD_C>': 0, '<UNK_C>': 1}
            for line in f:
                line = line.split('\t')
                if (line[0] == '\n'):
                    continue
                word_list = list(line[0])
                for char in word_list:
                    if (char not in vocab):
                        vocab.update({char: len(vocab)})
            return vocab
        
def get_prediction_labels(model, val_iterator, vocab):
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for id, batch in val_iterator:
            #padding then convert to tensor
            sentences = [sent['text'] for sent in batch]
            tags = [sent['tag'] for sent in batch]
            text = pad_sequence(sentences, batch_first=True, padding_value=vocab['word_to_idx']['<PAD>']).to(device)
            tag = pad_sequence(tags, batch_first=True, padding_value=vocab['tag_to_idx']['<PAD>']).to(device)
            predictions += model(text).cpu().numpy().tolist()
            labels += tag.cpu().numpy().tolist()
    return predictions, labels

def remove_mask(all_predictions, gold_predictions, mask):
    predicted, gold = [], []
    for i, row in enumerate(mask):
        for j, is_masked in enumerate(row):
            if is_masked:
                predicted.append(all_predictions[i][j])
                gold.append(gold_predictions[i][j])
    return predicted, gold
