from imports import *
from utils import *

class CharCNN(nn.Module):
    def __init__(self, alphabet_size, embedding_dim, hidden_dim, dropout):
        super(CharCNN, self).__init__()
        print("build char sequence feature extractor: CNN ...")
        self.hidden_dim = hidden_dim
        self.char_drop = nn.Dropout(dropout)
        self.char_embeddings = nn.Embedding(alphabet_size, embedding_dim)
        self.char_embeddings.weight.data.copy_(torch.from_numpy(CharCNN.random_embedding(alphabet_size, embedding_dim)))
        self.char_cnn = nn.Conv1d(embedding_dim, self.hidden_dim, kernel_size=3, padding=1)

    @staticmethod
    def random_embedding(vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        pretrain_emb[0, :] = np.zeros((1, embedding_dim))
        return pretrain_emb

    def forward(self, input):

        batch_size = input.size(0)
        char_embeds = self.char_drop(self.char_embeddings(input))
        char_embeds = char_embeds.transpose(2, 1).contiguous()
        char_cnn_out = self.char_cnn(char_embeds)
        char_cnn_out = F.max_pool1d(char_cnn_out, char_cnn_out.size(2)).contiguous().view(batch_size, -1)
        return char_cnn_out


class NeuralNet(nn.Module):

    def __init__(self, vocab, alphabet_size, tag_num):
        super(NeuralNet, self).__init__()
        self.use_char = USE_CHAR
        self.drop = nn.Dropout(DROPOUT)
        self.input_dim = WORD_EMBEDDING_DIM
        self.vocab_size = len(vocab['word_to_idx'])
        self.vocab = vocab
        self.embeds = nn.Embedding(self.vocab_size, WORD_EMBEDDING_DIM, padding_idx=0)
        
        self.embeds.weight.data.copy_(torch.from_numpy(self.random_embedding(self.vocab_size, WORD_EMBEDDING_DIM)))

        if self.use_char:
            self.input_dim += CHAR_HIDDEN_DIM
            self.char_feature = CharCNN(alphabet_size, CHAR_EMBEDDING_DIM, CHAR_HIDDEN_DIM, DROPOUT)

        self.lstm = nn.LSTM(self.input_dim, WORD_HIDDEN_DIM, batch_first=True, bidirectional=True)
        
        self.hidden2tag = nn.Linear(WORD_HIDDEN_DIM * 2, tag_num)

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(1, vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def neg_log_likelihood_loss(self, word_inputs, word_seq_lengths, char_inputs, batch_label, mask):
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        word_embeding = self.embeds(word_inputs)
        word_list = [word_embeding]
        if self.use_char:
            char_features = self.char_feature(char_inputs).contiguous().view(batch_size, seq_len, -1)
            word_list.append(char_features)
        word_embeding = torch.cat(word_list, 2)
        word_represents = self.drop(word_embeding)
        packed_words = pack_padded_sequence(word_represents, word_seq_lengths, batch_first=True, enforce_sorted=False)
        hidden = None
        lstm_out, hidden = self.lstm(packed_words, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out)
        lstm_out = lstm_out.transpose(0, 1)
        feature_out = self.drop(lstm_out)

        feature_out = self.hidden2tag(feature_out)
        loss_function = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
        feature_out = feature_out.contiguous().view(batch_size * seq_len, -1)
        total_loss = loss_function(feature_out, batch_label.contiguous().view(batch_size * seq_len))
        return total_loss

    def forward(self, word_inputs, word_seq_lengths, char_inputs, batch_label, mask):
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        word_embeding = self.embeds(word_inputs)
        word_list = [word_embeding]
        if self.use_char:
            char_features = self.char_feature(char_inputs).contiguous().view(batch_size, seq_len, -1)
            word_list.append(char_features)
        word_embeding = torch.cat(word_list, 2)
        word_represents = self.drop(word_embeding)
        packed_words = pack_padded_sequence(word_represents, word_seq_lengths, batch_first=True, enforce_sorted=False)
        hidden = None
        lstm_out, hidden = self.lstm(packed_words, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out)
        lstm_out = lstm_out.transpose(0, 1)
        feature_out = self.drop(lstm_out)
        feature_out = self.hidden2tag(feature_out)
        feature_out = feature_out.contiguous().view(batch_size * seq_len, -1)
        _, tag_seq = torch.max(feature_out, 1)
        tag_seq = tag_seq.view(batch_size, seq_len)
        mask = mask.to(device)
        tag_seq = mask.long() * tag_seq
        return tag_seq
    def predict(self, val_iterator, vocab):
        self.eval()
        val_iterator.create_batches()
        all_predictions = []
        save_file = open('preds.txt', 'w')
        num_sentences = 0
        for batchid, batch in enumerate(val_iterator.batches):
        # for batch in val_iterator.batches:
            batch_len = len(batch)
            sents = [sent['text'] for sent in batch]
            tags = [sent['tags'] for sent in batch]
            chars = [sent['chars'] for sent in batch]
            word_seq_lengths = [len(sent) for sent in sents]
            sents = pad_sequence(sents, batch_first=True, padding_value=self.vocab['word_to_idx']['<PAD_W>']).to(device)
            tags = pad_sequence(tags, batch_first=True, padding_value=self.vocab['tag_to_idx']['<PAD_T>']).to(device)
            chars = pad_sequence(chars, batch_first=True, padding_value=self.vocab['char_to_idx']['<PAD_C>']).to(device)
            mask = get_mask(sents)
            predictions = None
            with torch.no_grad():
                predictions = self.forward(sents, word_seq_lengths, chars, tags, mask)
            predictions = predictions.cpu().numpy().tolist()
            for i in range(batch_len):
                all_predictions.extend(predictions[i][:word_seq_lengths[i]])
        all_predictions = list(map(lambda x: vocab['idx_to_tag'][x], all_predictions))
        return all_predictions





class SimpleNet(nn.Module):
    def __init__(self, vocab_size, output_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, WORD_EMBEDDING_DIM)
        self.lstm = nn.LSTM(WORD_EMBEDDING_DIM, WORD_HIDDEN_DIM, batch_first=True, bidirectional=True)
        self.dropout_layer = nn.Dropout(DROPOUT)
        self.linear = nn.Linear(WORD_HIDDEN_DIM * 2, output_dim)

    def forward(self, batch, batch_len):
        embeddings = self.embeddings(batch)
        packed = pack_padded_sequence(embeddings, batch_len, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        dropout_out = self.dropout_layer(lstm_out)
        linear_out = self.linear(dropout_out)
        return linear_out

    def evaluate(self, val_iterator, labels):
        self.eval()
        val_iterator.create_batches()
        all_predictions = []
        gold_predictions = []
        for batchid, batches in enumerate(val_iterator.batches):
            sents = [sent['text'] for sent in batches]
            tags = [sent['tags'] for sent in batches]
            mask = get_mask(sents, self.vocab).to(device)
            sents = pad_sequence(sents, batch_first=True, padding_value=self.vocab['word_to_idx']['<PAD_W>']).to(device)
            tags = pad_sequence(tags, batch_first=True, padding_value=self.vocab['tag_to_idx']['<PAD_W>']).to(device)
            batch_len = [len(sent) for sent in batches]
            predictions = self.forward(sents, batch_len)
            all_predictions.extend(predictions)
            gold_predictions.extend(tags)
        all_predictions, gold_predictions = remove_mask(all_predictions, gold_predictions, mask)
        #compute f1 macro and f1 micro score 
        f1_macro = f1_score(gold_predictions, all_predictions, average='macro')
        f1_micro = f1_score(gold_predictions, all_predictions, average='micro')
        print("Validation F1 Micro: {}, F1 Macro {}, Avg = {}".format(f1_micro, f1_macro, (f1_micro + f1_macro) / 2))


