from imports import *
#parse the command line arguments
# nltk.download('averaged_perceptron_tagger')
# nltk.download('vader_lexicon')
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
    parser.add_argument('--min_df', type=int, default=1, help='min_df to use for the model')
    parser.add_argument('--max_df', type=float, default=1.0, help='max_df to use for the model')
    parser.add_argument('--tfidf_max_feat', type=int, default=None, help='max_features to use for the model')
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
        self.sid = SentimentIntensityAnalyzer()
        try:
            self.stopwords = set(stopwords.words("english"))
        except:
            nltk.download('stopwords')
            self.stopwords = set(stopwords.words("english"))

    def preprocess(self):
        
        self.train_df["sentiment"] = self.train_df["reviews"].apply(lambda x: self.sid.polarity_scores(x))  
        self.train_df["reviews"] = self.train_df["reviews"].apply(self.clean)
        self.train_df = pd.concat([self.train_df.drop(['sentiment'], axis=1), self.train_df['sentiment'].apply(pd.Series)], axis=1) 
        self.train_df["word_count"] = self.train_df["reviews"].apply(lambda x: len(str(x).split()))
        self.train_df["char_count"] = self.train_df["reviews"].apply(lambda x: sum(len(word) for word in str(x).split()))
        tfidf =TfidfVectorizer(min_df = self.args.min_df, max_df=self.args.max_df, use_idf=True, norm = "l2",ngram_range=(1,self.args.ngrams), sublinear_tf=True, max_features=self.args.tfidf_max_feat)
        tfidf_result = tfidf.fit_transform(self.train_df["reviews"]).toarray()
        tfidf_df = pd.DataFrame(tfidf_result, columns = tfidf.get_feature_names_out())
        tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
        tfidf_df.index = train_df.index
        self.train_df = pd.concat([self.train_df, tfidf_df], axis=1)
        print(self.train_df.head(5))
        return self.train_df
    
    def clean(self, text):
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
        text = text.lower()
        words = text.split()
        words = [word.strip(string.punctuation) for word in words if not any(c.isdigit() for c in word) and word not in self.stopwords]
        words = [word for word in words if len(word) > 2]
        pos_tags = nltk.pos_tag(words)
        words = [self.lemmatizer.lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags if len(t[0]) > 1]
        text = " ".join(words)
        return text

class DataLoader:
    def __init__(self, args):
        self.args = args

    def load_data(self):
        df = pd.read_csv(self.args.train_file, header=None, names=['reviews', 'labels'], nrows=1000)
        df.dropna(axis = 0, inplace=True)
        return df
    
    def test_train_split(self, train_df, test_size=0.2):
        ignore_features = ['reviews', 'labels']
        features = [x for x in train_df.columns if x not in ignore_features]
        X_train, X_test, y_train, y_test = train_test_split(train_df[features], train_df['labels'], test_size=self.args.test_size, random_state=self.args.random_state)
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

        # self.tfidf_vectorizer = TfidfVectorizer(max_df=0.95, use_idf=1, norm = "l2", smooth_idf=1, sublinear_tf=1)
        # self.X_train_tfidf = self.tfidf_vectorizer.fit_transform(self.X_train)
        # self.X_test_tfidf = self.tfidf_vectorizer.transform(self.X_test)

        print(self.X_train.shape, self.y_train.shape)
        grid_params = {
            'C': [0.1, 1, 10, 100], 
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'] 
            }
        scoring = make_scorer(lambda x,y : (f1_score(x,y,average='micro')+f1_score(x,y,average='macro'))/2, greater_is_better=True)
        if (args.model == 'nb'): 
            self.model = MultinomialNB()
        elif (args.model == 'lr'): 
            self.model = LogisticRegression(C=1.0, class_weight='balanced', solver='liblinear', multi_class='ovr')
        elif (args.model == 'rf'): 
            self.model = RandomForestClassifier(n_estimators=100)
        elif (args.model == 'svm'): 
            self.model = LinearSVC()
        elif (args.model == 'knn'): 
            self.model = KNeighborsClassifier()
        elif (args.model == 'grid_lr'): 
            self.model = GridSearchCV(LogisticRegression(solver='liblinear'),
                                        scoring=scoring,
                                        param_grid = grid_params, cv=5, verbose=True, n_jobs=-1)
        else: raise Exception('Invalid model')
    
    def train(self):
        self.model.fit(self.X_train, self.y_train, sample_weight=self.sample_weights)

    def predict(self):
        self.y_score = self.model.decision_function(self.X_test)
        self.y_pred = self.model.predict(self.X_test)

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
    Loader = DataLoader(args)
    train_df = Loader.load_data()
    print(args)
    text_preprocessor = TextPreprocessor(train_df, args)
    train_df = text_preprocessor.preprocess()

    print(train_df.head(5))
    X_train, X_test, y_train, y_test, sample_weights = Loader.test_train_split(train_df, args.test_size)
    print(X_train.head(10))

    classifier = Classifier(args, X_train, X_test, y_train, y_test, sample_weights)
    classifier.train()
    classifier.predict()
    classifier.evaluate()

