from imports import *
from utils import SentenceLoader, WordVocabulary, TagVocabulary, CharVocabulary
from dataset import NERDataset
from models import NeuralNet, SimpleNet
from eval import Evaluator

def train_epoch(epoch, train_iterator, vocab, model, optimizer, criterion):
    train_iterator.create_batches()
    model.train()
    # for batch_id, batch in enumerate(train_iterator.batches):
    total_loss = 0
    num_batches = len(train_iterator)
    with tqdm(total=num_batches, desc="Training", unit="batch", leave=False) as pbar:
        for _, batch in enumerate(train_iterator.batches):
            optimizer.zero_grad()
            #use pad packed sequence
            sents = [sent['text'] for sent in batch]
            tags = [sent['tags'] for sent in batch]
            chars = [sent['chars'] for sent in batch]
            sent_lengths = [len(sent) for sent in sents]
            curr_batch_size = len(sents)
            sents = pad_sequence(sents, batch_first=True, padding_value=vocab['word_to_idx']['<PAD_W>']).to(device)
            tags = pad_sequence(tags, batch_first=True, padding_value=vocab['tag_to_idx']['<PAD_T>']).to(device)
            chars = pad_sequence(chars, batch_first=True, padding_value=vocab['char_to_idx']['<PAD_C>']).to(device)
            mask = get_mask(sents)
            loss = model.neg_log_likelihood_loss(sents, sent_lengths, chars, tags, mask)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            pbar.set_description(f"Epoch: {epoch}")
            pbar.update(1)
        return total_loss
def train(model, train_iterator, val_iterator, vocab, val_path):
    optimizer = optim.Adam(model.parameters(), lr=INIT_LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['word_to_idx']['<PAD_W>'])
    evaluator = Evaluator(val_path)
    print("Starting Training")
    best_so_far = 0
    for epoch in range(EPOCHS):
        train_loss = train_epoch(epoch+1, train_iterator, vocab, model, optimizer, criterion)
        predictions = model.predict(val_iterator, vocab)
        micro_f1, macro_f1 = evaluator.evaluate_predictions(predictions)
        if (micro_f1+macro_f1)/2 > best_so_far:
            best_so_far = (micro_f1+macro_f1)/2
            with open(SAVE_FILE, 'wb') as f:
                pickle.dump({"model": model, "vocab": vocab}, f)
        print("Epoch:", epoch+1, "Completed, Avg train Loss:", round(train_loss/len(train_iterator), 2), "Val avg F1", round(100 * (micro_f1+macro_f1)/2, 2))
if __name__ == "__main__":
    train_path, val_path = "data/train.txt", "data/dev.txt"
    vocab = {
                "word_to_idx": WordVocabulary(train_path).vocab,
                "tag_to_idx": TagVocabulary(train_path).vocab, 
                "char_to_idx": CharVocabulary(train_path).vocab
            }
    vocab['idx_to_tag'] = {v: k for k, v in vocab['tag_to_idx'].items()}

    alphabet_size = len(vocab['char_to_idx'])
    num_tags = len(vocab['tag_to_idx'])

    #word_to_idx has <PAD_W> and <UNK_W> as well
    
    train_dataset = NERDataset(vocab, train_path, "Train")
    val_dataset = NERDataset(vocab, val_path, "Val")
    #minimum and maximum length of sentences in the dataset

    train_iterator = BucketIterator(train_dataset, batch_size=TRAIN_BATCH_SIZE, sort_key=lambda x: len(x['text']), repeat=True, shuffle=True, sort = False, device=device, sort_within_batch=True)
    val_iterator = BucketIterator(val_dataset, batch_size=VAL_BATCH_SIZE, sort_key=lambda x: len(x['text']), device=device, shuffle=False)
    # val_iterator = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, device=device)

    print('Created `train_iterator` with %d batches!'%len(train_iterator))
    print('Created `val_iterator` with %d batches!'%len(val_iterator))
    
    print("Current Device set to: ", device)

    model = NeuralNet(vocab, alphabet_size, num_tags).to(device)
    # micro_f1, macro_f1 = model.evaluate(val_iterator, vocab)
    # print("Micro F1: ", micro_f1, "Macro F1: ", macro_f1)
    train(model, train_iterator, val_iterator, vocab, val_path)
    # torch.load(model.state_dict(), SAVE_FILE)



