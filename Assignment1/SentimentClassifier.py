import numpy as np
import pandas as pd
from sklearn_deltatfidf import DeltaTfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

class SentimentClassifier:
    def __init__(self, model_name,vectorizer_name, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.vectorizer_name = vectorizer_name
        if vectorizer_name == "tf_idf" :
          self.vectorizer = TfidfVectorizer(analyzer="word", norm="l1", use_idf=True, dtype=np.float32,ngram_range=(1, 3), min_df=5)
        elif vectorizer_name == "delta_tf_idf" :
          self.vectorizer = DeltaTfidfVectorizer(analyzer="word", norm="l1", use_idf=True, dtype=np.float32,ngram_range=(1, 3), min_df=5)

        self.model = None

    def train(self, X_train, y_train):

        # Vectorizer
        if self.vectorizer_name == "tf_idf" :
          X_train = self.vectorizer.fit_transform(X_train)
        elif self.vectorizer_name == "delta_tf_idf" :
          X_train = self.vectorizer.fit_transform(X_train.tolist(), y_train.tolist())

        # Model
        if self.model_name == "naive_bayes_gaussian" :
          self.model = GaussianNB(**self.kwargs)

        elif self.model_name == "naive_bayes_multinomial" :
          self.model = MultinomialNB(**self.kwargs)

        elif self.model_name == "decision_tree" :
          self.model = DecisionTreeClassifier(**self.kwargs)

        elif self.model_name == "random_forest" :
          self.model = RandomForestClassifier(**self.kwargs)

        # Training
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        X_test = self.vectorizer.transform(X_test.tolist())
        return self.model.predict(X_test)
