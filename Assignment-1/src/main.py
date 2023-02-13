from imports import *

warnings.filterwarnings("ignore", category=FutureWarning)
#parse the command line arguments
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
        
def parse_args():
    parser = argparse.ArgumentParser(description='arguements for the text classifier')
    parser.add_argument('--train_file', type=str, default='train.csv', help='path to training data')
    parser.add_argument('--model_path', type=str, default='model.pkl', help='path to save the model')
    parser.add_argument('--operation', type=str, default='all', help='operation to perform', choices=['train', 'test', 'all'])
    parser.add_argument('--lemmatize', type=int, default=1, help='lemmatize the text')
    parser.add_argument('--stem', type=int, default=0, help='stem the text')
    parser.add_argument('--remove_stopwords', type=int, default=1, help='remove stop words')
    parser.add_argument('--random_state', type=int, default=42, help='random state for train test split')
    parser.add_argument('--test_size', type=float, default=0.2, help='test size for train test split')
    parser.add_argument('--model', type=str, default='lr', help='model to use', choices=['nb', 'lr', 'rf', 'svm', 'grid_lr'])
    parser.add_argument('--ngrams', type=int, default=2, help='ngrams to use for the model')
    parser.add_argument('--min_df', type=int, default=5, help='min_df to use for the model')
    parser.add_argument('--max_df', type=float, default=0.75, help='max_df to use for the model')
    parser.add_argument('--tfidf_max_feat', type=int, default=100000, help='max_features to use for the model')
    parser.add_argument('--test_file', type=str, default='test.csv', help='path to test data')
    parser.add_argument('--outfile', type=str, default='output.csv', help='path to save predicted label' )
    parser.add_argument('--tfidf_file', type=str, default='tfidf.pkl', help='path to save tfidf vectorizer' )
    return parser

class TextPreprocessor():
    def __init__(self, train_df, args):
        self.args = args
        self.train_df = train_df
        self.lemmatize = args.lemmatize
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stem = args.stem
        self.remove_stopwords = args.remove_stopwords
        self.stopwords = set(stopwords.words("english"))

    def preprocess(self):
        if (self.lemmatize): self.lemmatize_text(self.train_df)
        if (self.stem): self.stem_text(self.train_df)
        if (self.remove_stopwords): self.remove_stopwords_(self.train_df)
        self.train_df["reviews"] = self.train_df["reviews"].str.replace('[{}]'.format(string.punctuation), '')
        if (args.operation == 'test'):
            return self.train_df["reviews"]
        else:
            return self.train_df["reviews"], self.train_df["labels"]
    def lemmatize_text(self, train_df):
        train_df["reviews"] = train_df["reviews"].apply(lambda x: " ".join([self.lemmatizer.lemmatize(word) for word in x.split()]))
        print("Lemmatization completed")
    def stem_text(self, train_df):
        train_df["reviews"] = train_df["reviews"].apply(lambda x: " ".join([self.stemmer.stem(word) for word in x.split()]))
        print("Stemming completed")
    def remove_stopwords_(self, train_df):
        train_df["reviews"] = train_df["reviews"].apply(lambda x: " ".join([word for word in x.split() if word not in self.stopwords]))
        print("Stopwords Removed")
    
    # def lemmatize_and_postag(self, text):
    #     words = text.split()
    #     pos_tags = nltk.pos_tag(words)
    #     words = [self.lemmatizer.lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags if len(t[0]) > 1]
    #     return " ".join(words)
class DataLoader:
    def __init__(self, args):
        self.args = args

    def load_data(self):
        df = pd.read_csv(self.args.train_file, header=None, names=['reviews', 'labels'])
        df.dropna(axis = 0, inplace=True)
        return df
    
    def load_test(self):
        df = pd.read_csv(self.args.test_file, header=None, names=['reviews'])
        return df
    def test_train_split(self, X, y, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.args.test_size, random_state=self.args.random_state)
        return X_train, X_test, y_train, y_test

class Classifier:
    def __init__(self, args):
        self.args = args
        grid_params = {
            'C': [0.1, 1, 10, 20],
            'penalty': ['l2', 'elasticnet'],
            'class_weight': ['balanced'],
            'solver': ['liblinear', 'saga', 'lbfgs', 'newton-cg']
        }
        scoring = make_scorer(lambda x,y : (f1_score(x,y,average='micro')+f1_score(x,y,average='macro'))/2, greater_is_better=True)
        if (args.model == 'nb'): self.model = MultinomialNB()
        elif (args.model == 'lr'): self.model = LogisticRegression(C=1, class_weight='balanced', solver='liblinear', multi_class='ovr', max_iter = 1000, verbose=False)
        elif (args.model == 'rf'): self.model = RandomForestClassifier()
        elif (args.model == 'svm'): self.model = LinearSVC()
        elif (args.model == 'grid_lr'): 
            self.model = GridSearchCV(LogisticRegression(multi_class='ovr'), 
                                        scoring=scoring,
                                        param_grid = grid_params, cv=5, verbose=True, n_jobs=16)
        else: raise Exception('Invalid model')
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred

    def evaluate(self, y_pred, y_test):
        predicted_categories = y_pred
        predicted_categories = pd.Series(predicted_categories)
        f1_micro = f1_score(y_test, predicted_categories, average='micro')
        f1_macro = f1_score(y_test, predicted_categories, average='macro')
        print("F1 Score on single fold = {}%".format(100.0 * (f1_micro+f1_macro)/2.0))
        if (self.args.model == 'grid_lr'):
            print("Best params: ", self.model.best_params_)
            print("Best score: ", self.model.best_score_)
            print("Best estimator: ", self.model.best_estimator_)
        return 100.0 * (f1_micro+f1_macro)/2.0
if __name__ == '__main__':
    args = parse_args().parse_args()
    if (args.operation == 'all'):

        Loader = DataLoader(args)
        train_df = Loader.load_data()

        text_preprocessor = TextPreprocessor(train_df, args)
        X, y = text_preprocessor.preprocess()
        print("Pre processing done")

        kfold = KFold(n_splits=5, shuffle=True, random_state=args.random_state)

        tfidf = TfidfVectorizer(min_df = args.min_df, max_features = args.tfidf_max_feat, ngram_range=(1,args.ngrams), use_idf=True, norm='l2', smooth_idf=True, sublinear_tf=True)
        tfidf.fit(X)
        scores = 0.0
        def run_fold(train_index, test_index):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            X_train = tfidf.transform(X_train)
            X_test = tfidf.transform(X_test)
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            classifier = Classifier(args)
            model_trained = classifier.train(X_train, y_train)
            y_pred = classifier.predict(X_test)
            score = classifier.evaluate(y_pred, y_test)
            return score

        folds = [(train_index, test_index) for train_index, test_index in kfold.split(X)]
        pool = multiprocessing.Pool()
        results = pool.starmap(run_fold, folds)

        pool.close()
        pool.join()
        print("F1 score (K-fold) = ", np.mean(results))
        print(args)

    if (args.operation == 'train'):
        Loader = DataLoader(args)
        train_df = Loader.load_data()

        text_preprocessor = TextPreprocessor(train_df, args)
        X, y = text_preprocessor.preprocess()
        print("Pre processing done")

        tfidf = TfidfVectorizer(min_df = args.min_df, max_features = args.tfidf_max_feat, ngram_range=(1,args.ngrams), use_idf=True, norm='l2', smooth_idf=True, sublinear_tf=True)
        tfidf.fit(X)
        with open(args.tfidf_file, 'wb') as tfidf_filename:
            pickle.dump(tfidf, tfidf_filename)
        X = tfidf.transform(X)
        classifier = Classifier(args)
        classifier.train(X, y)
        with open(args.model_path, 'wb') as model_file:
            pickle.dump(classifier.model, model_file)

        print("Model saved to ", args.model_path)
    if (args.operation == 'test'):
        Loader = DataLoader(args)
        test_df = Loader.load_test()
        text_preprocessor = TextPreprocessor(test_df, args)
        X = text_preprocessor.preprocess()
        print("Pre processing done")

        tfidf = pickle.load(open(args.tfidf_file, 'rb'))
        X = tfidf.transform(X)
        model_trained = None
        try:
            with open(args.model_path, 'rb') as model_file:
                model_trained = pickle.load(model_file)
        except:
            exit(0)
        y_pred = model_trained.predict(X)
        df = pd.DataFrame(y_pred)
        df.to_csv(args.outfile, index=False, header = None)
        print("Predictions saved to ", args.outfile)





