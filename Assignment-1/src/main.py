from imports import *

#parse the command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='arguements for the text classifier')
    parser.add_argument('--train_file', type=str, default='train.csv', help='path to training data')
    parser.add_argument('--lemmatize', type=int, default=True, help='lemmatize the text')
    parser.add_argument('--stem', type=int, default=True, help='stem the text')
    parser.add_argument('--remove_stopwords', type=int, default=True, help='remove stop words')
    parser.add_argument('--out_file', type=str, default='train_processed.csv', help='path to processed training data')
    parser.add_argument('--random_state', type=int, default=42, help='random state for train test split')
    parser.add_argument('--test_size', type=float, default=0.2, help='test size for train test split')
    parser.add_argument('--model', type=str, default='nb', help='model to use', choices=['nb', 'lr', 'rf', 'svm', 'knn', 'grid_lr'])
    parser.add_argument('--ngrams', type=int, default=1, help='ngrams to use for the model')
    parser.add_argument('--min_df', type=float, default=1, help='min_df to use for the model')
    parser.add_argument('--max_df', type=float, default=1.0, help='max_df to use for the model')
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

        weights = pd.Series(y_train).value_counts().reset_index()
        weights.columns = ['label', 'counts']
        weights['weights'] = weights['counts'].sum() / weights['counts']
        weights = weights.set_index('label')['weights'].to_dict()
        sample_weights = pd.Series(y_train).map(weights)

        return X_train, X_test, y_train, y_test, sample_weights

class Classifier:
    def __init__(self, args, X_train, X_test, y_train, y_test, sample_weights):
        self.args = args
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.sample_weights = sample_weights
        self.tfidf_vectorizer = TfidfVectorizer(min_df=args.min_df, max_df=args.max_df, ngram_range=(1,2), max_features=1000, sublinear_tf=True, norm='l2', use_idf=True, smooth_idf=True)
        self.X_train_tfidf = self.tfidf_vectorizer.fit_transform(self.X_train)
        self.X_test_tfidf = self.tfidf_vectorizer.transform(self.X_test)
        # print("Sampling starting")
        # print(self.X_train_tfidf.shape, self.y_train.shape, self.y_train.value_counts())
        #Solve class imbalance
        # smote = SMOTE(sampling_strategy='auto', random_state=42)
        # self.X_train_balanced, self.y_train_balanced = smote.fit_resample(self.X_train_tfidf, self.y_train)

        # smote_enn = SMOTEENN(sampling_strategy='auto', random_state=0, n_jobs=-1)
        # self.X_train_balanced, self.y_train_balanced = smote_enn.fit_resample(self.X_train_tfidf, self.y_train)

        # print("Finished starting")
        # print(self.X_train_balanced.shape, self.y_train_balanced.shape, self.y_train_balanced.value_counts())
        grid_params = {
            'C': [0.1, 1, 10], 
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear'] 
            }
        scoring = make_scorer(lambda x,y : (f1_score(x,y,average='micro')+f1_score(x,y,average='macro'))/2, greater_is_better=True)
        if (args.model == 'nb'): self.model = MultinomialNB()
        elif (args.model == 'lr'): self.model = LogisticRegression(C = 1, solver='liblinear', penalty="l2")
        elif (args.model == 'rf'): self.model = RandomForestClassifier()
        elif (args.model == 'svm'): self.model = LinearSVC()
        elif (args.model == 'knn'): self.model = KNeighborsClassifier()
        elif (args.model == 'grid_lr'): 
            self.model = GridSearchCV(LogisticRegression(solver='liblinear'), 
                                        scoring=scoring,
                                        param_grid = grid_params, cv=5, verbose=True, n_jobs=-1)
        else: raise Exception('Invalid model')
    
    def train(self):
        self.model.fit(self.X_train_tfidf, self.y_train, sample_weight=self.sample_weights)

    def predict(self):
        self.y_pred = self.model.predict(self.X_test_tfidf)

    def evaluate(self):
        predicted_categories = self.y_pred
        predicted_categories = pd.Series(predicted_categories)
        f1_micro = f1_score(y_test, predicted_categories, average='micro')
        f1_macro = f1_score(y_test, predicted_categories, average='macro')
        print("F1 Score = {}%".format(100.0 * (f1_micro+f1_macro)/2.0))
        if (self.args.model == 'grid_lr'):
            print("Best params: ", self.model.best_params_)
            print("Best score: ", self.model.best_score_)
            print("Best estimator: ", self.model.best_estimator_)
        print("Params: ", self.args)
    
if __name__ == '__main__':
    args = parse_args().parse_args()
    print(args)
    # Loader = DataLoader(args)
    # train_df = Loader.load_data()

    # text_preprocessor = TextPreprocessor(train_df, args)
    # # text_preprocessor.preprocess()
    # train_df.to_csv(args.out_file, index=False)

    # print(train_df.head(5))
    # X_train, X_test, y_train, y_test, sample_weights = Loader.test_train_split(train_df, args.test_size)
    # classifier = Classifier(args, X_train, X_test, y_train, y_test, sample_weights)
    # classifier.train()
    # classifier.predict()
    # classifier.evaluate()
