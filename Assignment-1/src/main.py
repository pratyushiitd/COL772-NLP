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
    parser.add_argument('--model', type=str, default='nb', help='model to use', choices=['nb', 'lr', 'rf', 'svm', 'knn'])
    parser.add_argument('--ngrams', type=int, default=1, help='ngrams to use for the model')
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
    def __init__(self, args):
        self.args = args

    def load_data(self):
        df = pd.read_csv(self.args.train_file, header=None, names=['reviews', 'labels'])
        df.dropna(axis = 0, inplace=True)
        return df
    
    def test_train_split(self, train_df, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(train_df['reviews'], train_df['labels'], test_size=self.args.test_size, random_state=self.args.random_state)
        return X_train, X_test, y_train, y_test

class Classifier:
    def __init__(self, args, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        if (args.model == 'nb'): self.model = MultinomialNB(class_weight='balanced')
        elif (args.model == 'lr'): self.model = LogisticRegression()
        elif (args.model == 'rf'): self.model = RandomForestClassifier()
        elif (args.model == 'svm'): self.model = LinearSVC()
        elif (args.model == 'knn'): self.model = KNeighborsClassifier()
        else: raise Exception('Invalid model')
        self.Pipeline = Pipeline([
                ('vect', CountVectorizer(ngram_range=(1, args.ngrams))),
                ('tfidf', TfidfTransformer()),
                ('clf', self.model)])
        
    def train(self):
        self.Pipeline.fit(self.X_train, self.y_train, clf__sample_weight=None)

    def predict(self):
        self.y_pred = self.Pipeline.predict(self.X_test)

    def evaluate(self):
        predicted_categories = self.y_pred
        predicted_categories = pd.Series(predicted_categories)
        f1_micro = f1_score(y_test, predicted_categories, average='micro')
        f1_macro = f1_score(y_test, predicted_categories, average='macro')
        print("F1 Score = ", (f1_micro+f1_macro)/2)


if __name__ == '__main__':
    args = parse_args().parse_args()
    Loader = DataLoader(args)
    train_df = Loader.load_data()

    # text_preprocessor = TextPreprocessor(train_df, args)
    # text_preprocessor.preprocess()
    # train_df.to_csv(args.out_file, index=False)

    print(train_df.head)
    X_train, X_test, y_train, y_test = Loader.test_train_split(train_df, args.test_size)
    classifier = Classifier(args, X_train, X_test, y_train, y_test)
    classifier.train()
    classifier.predict()
    classifier.evaluate()

