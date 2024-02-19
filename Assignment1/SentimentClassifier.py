import numpy as np
import pandas as pd
from sklearn_deltatfidf import DeltaTfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from joblib import load,dump
from chisquare_feature_selection import chi_square_tfidf_feature_selection

class SentimentClassifier:
    def __init__(self, model_name,vectorizer_name,feature_selection, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.vectorizer_name = vectorizer_name
        self.model_path = f"/content/{self.kwargs}_model.pt"
        self.vectorizer_path = f"/content/{self.vectorizer_name}_vectorizer.pt"
        if vectorizer_name == "tf_idf" :
          self.vectorizer = TfidfVectorizer(analyzer="word", norm="l1", use_idf=True, dtype=np.float32,ngram_range=(1, 3), min_df=5)
        elif vectorizer_name == "delta_tf_idf" :
          self.vectorizer = DeltaTfidfVectorizer(analyzer="word", norm="l1", use_idf=True, dtype=np.float32,ngram_range=(1, 3), min_df=5)

        self.model = None
        self.feature_selection = feature_selection
        self.test_pred_path = f"/content/{self.kwargs}_pred.csv"

    def train(self, X_train, y_train):

        # Vectorizer
        if self.vectorizer_name == "tf_idf" :
          X_train_tfidf = self.vectorizer.fit_transform(X_train)
        elif self.vectorizer_name == "delta_tf_idf" :
          X_train_tfidf = self.vectorizer.fit_transform(X_train.tolist(), y_train.tolist())

        # Feature Selection
        if self.feature_selection :
          print("Running Feature Selection ...")
          self.important_feature_indices = chi_square_tfidf_feature_selection(X_train_tfidf, y_train, k=int(0.4*len(self.vectorizer.get_feature_names_out())))
          feature_names = self.vectorizer.get_feature_names_out()[self.important_feature_indices]
          X_train_tfidf = self.vectorizer.transform(X_train)[:, self.important_feature_indices]

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
        if self.model_name in ['naive_bayes_gaussian', 'naive_bayes_multinomial']:
          X_train_tfidf = np.asarray(X_train_tfidf.todense())
        self.model.fit(X_train_tfidf, y_train)

        #save model 
        dump(self.model, self.model_path)

        #save vectorizer
        dump(self.vectorizer, self.vectorizer_path )

    def predict(self, X_test, y_test):
        vectorizer = load(self.vectorizer_path)
        model = load(self.model_path)

        if self.feature_selection :
          X_test_tf_idf = vectorizer.transform(X_test.tolist())[:, self.important_feature_indices]
        else :
          X_test_tf_idf = vectorizer.transform(X_test.tolist())

        if self.model_name in ['naive_bayes_gaussian', 'naive_bayes_multinomial']:
          X_test_tf_idf = np.asarray(X_test_tf_idf.todense())
        y_pred = model.predict(X_test_tf_idf)
        df_pred = pd.DataFrame({"Review": X_test.tolist(), "Actual": y_test, "Predicted": y_pred})
        df_pred.to_csv(self.test_pred_path, index=False)
        return y_pred
