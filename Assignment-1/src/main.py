from imports import *

#parse the command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='arguements for the text classifier')
    parser.add_argument('--train_file', type=str, default='train.csv', help='path to training data')
    parser.add_argument('--lemmatize', type=bool, default=True, help='lemmatize the text')
    parser.add_argument('--stem', type=bool, default=True, help='stem the text')
    parser.add_argument('--remove_stopwords', type=bool, default=True, help='remove stop words')
    parser.add_argument('--out_file', type=str, default='train_processed.csv', help='path to processed training data')
    parser.add_argument('--random_state', type=int, default=42, help='random state for train test split')
    parser.add_argument('--test_size', type=float, default=0.2, help='test size for train test split')
    return parser

class TextPreprocessor():
    def __init__(self, train_df, args):
        self.train_df = train_df
        self.lemmatize = args.lemmatize
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stem = args.stem
        self.remove_stopwords = args.remove_stopwords
        try:
            self.stopwords = set(stopwords.words("english"))
        except:
            nltk.download('stopwords')
            self.stopwords = set(stopwords.words("english"))
    
    def preprocess(self):
        if (self.lemmatize): self.lemmatize_text(self.train_df)
        print(self.train_df.head())
        if (self.stem): self.stem_text(self.train_df)
        print(self.train_df.head())
        if (self.remove_stopwords): self.remove_stopwords_(self.train_df)
        print(self.train_df.head())
    
    def lemmatize_text(self, train_df):
        train_df["reviews"] = train_df["reviews"].apply(lambda x: " ".join([self.lemmatizer.lemmatize(word) for word in x.split()]))

    def stem_text(self, train_df):
        train_df["reviews"] = train_df["reviews"].apply(lambda x: " ".join([self.stemmer.stem(word) for word in x.split()]))

    def remove_stopwords_(self, train_df):
        train_df["reviews"] = train_df["reviews"].apply(lambda x: " ".join([word for word in x.split() if word not in self.stopwords]))
    
class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        df = pd.read_csv(self.file_path, header=None, names=['reviews', 'labels'])
        df.dropna(axis = 0, inplace=True)
        return df
    
    def test_train_split(self, df, test_size=0.2):
        train, test = train_test_split(df, test_size=test_size, random_state=42)
        return train, test

class Classifier:
    def __init__(self, train_df, test_df):
        self.train_df = train_df
        self.test_df = test_df

    def train(self):
        
        pass

    def predict(self):
        pass

    def evaluate(self):
        pass

if __name__ == '__main__':
    args = parse_args().parse_args()
    train_df = DataLoader(args.train_file).load_data()
    print(train_df.head())

    # text_preprocessor = TextPreprocessor(train_df, args)
    # text_preprocessor.preprocess()
    # train_df.to_csv(args.out_file, index=False)

    train, test = DataLoader(args.train_file).test_train_split(train_df, args.test_size)

