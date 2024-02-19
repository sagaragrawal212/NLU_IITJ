import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn_deltatfidf import DeltaTfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from joblib import load,dump
from sklearn.feature_selection import chi2

#imports for preprocessing task 2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
# import string
import re
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text_task2(text):
    # Lowercasing
    text = text.lower()

    # Removing URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Removing emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    # Removing special characters, keeping letters numbers,
    # text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'[^\w\s]|_', '', text)

    # Removing punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenization
    tokens = word_tokenize(text)

    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    # Joining tokens back into a single string
    preprocessed_text = ' '.join(lemmatized_tokens)

    return preprocessed_text


def chi_square_tfidf_feature_selection(X_train_tfidf, y_train, k=1000):
  """
  Performs feature selection using Chi-square and TF-IDF.
  """

  # Perform Chi-square test
  chi2_scores, _ = chi2(X_train_tfidf, y_train)

  # Get indices of top k features
  feature_indices = chi2_scores.argsort()[-k:]

  return feature_indices

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
        return y_pred, df_pred

def multi_to_binary_transform(label):
  if label in [3,4,5]:
    return 1
  else:
    return 0

def evaluate_prediction_on_test_set(y_test, y_pred):
  accuracy = accuracy_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred, average = 'macro')
  recall = recall_score(y_test, y_pred, average = 'macro')
  f1 = f1_score(y_test, y_pred, average = 'macro')
  conf_matrix = confusion_matrix(y_test, y_pred)

  
  print("Accuracy:", accuracy)
  print("Precision:", precision)
  print("Recall:", recall)
  print("F1 Score:", f1)
  # print("Confusion Matrix:\n", conf_matrix)
  # return accuracy, precision, recall, f1, conf_matrix

data_path = "/content/AMAZON_FASHION.csv"
vectorizer = "delta_tf_idf"
feature_selection = True

if __name__ == '__main__':
  ## Read Data
  print("Loading Data ...")
  df = pd.read_csv(data_path)
  df = df[df.reviewText.notnull()]
  # X = df['reviewText']
  y = df['overall'].astype(int)
  
  #Data Preprocessing 
  df['reviewTextCLeaned'] = df['reviewText'].apply(lambda x : preprocess_text_task2(x))
  
  X = df['reviewTextCLeaned']
  
  # Perform test train split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
  # Assuming you have X_test and y_test as the test data and labels
  # Define the parameters for each model
  dt_params = ['criterion__entropy','criterion__gini']
  rf_params = ['n_estimators__20', 'n_estimators__50', 'n_estimators__100']
  gnb_params = ['var_smoothing__1e-9' ]
  mnb_params = ['alpha__1' ]
  
  # Define a list of classifiers with their respective parameters
  classifiers = [
      {'model_name': 'naive_bayes_gaussian', 'params': gnb_params},
       {'model_name': 'naive_bayes_multinomial', 'params': mnb_params},
      {'model_name': 'decision_tree', 'params': dt_params},
      {'model_name': 'random_forest', 'params': rf_params}
  ]
  
  print("Training Start ...")
  # Iterate over each classifier and its parameters
  for classifier in classifiers:
      model_name = classifier['model_name']
      params = classifier['params']
      print(f"Training {model_name} with params : {params}")
      for param in params :
  
          key,val = param.split("__")
          if key == 'var_smoothing':
            val = float(val)
          elif val.isnumeric() :
            val = int(val)
  
          param_dict = {key : val}
          # Instantiate the SentimentClassifier object
          clf = SentimentClassifier(model_name,vectorizer,feature_selection, **param_dict)
          
          # Train the classifier
          clf.train(X_train, y_train)
  
          # Make predictions on the test data
          pred, df_pred = clf.predict(X_test, y_test)
  
          # Calculate evaluation metrics
          accuracy = accuracy_score(y_test, pred)
          # precision = precision_score(y_test, pred)
          # recall = recall_score(y_test, pred)
          # f1score = f1_score(y_test, pred)
          # confusion = confusion_matrix(y_test, pred)
  
          # Print the results
          print(f"Results for {model_name} with parameters {param_dict}:")
          print("Accuracy:", accuracy)
          # print("Precision:", precision)
          # print("Recall:", recall)
          # print("F1 Score:", f1score)
          # print("Confusion Matrix:\n", confusion)
          
          #Create a confusion matrix
          labels = sorted(list(set(df_pred['Actual']).union(set(df_pred['Predicted']))))
          cm = confusion_matrix(df_pred['Actual'], df_pred['Predicted'], labels=labels)
        
          # Plot confusion matrix with annotations
          plt.figure(figsize=(10, 8))
          sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
          plt.xlabel('Predicted')
          plt.ylabel('Actual')
          plt.title('Confusion Matrix For Multi class Classification')
          plt.show()

          # binary evaluation
          df_pred['binary_true'] = df_pred.Actual.apply(multi_to_binary_transform)
          df_pred['binary_pred'] = df_pred.Predicted.apply(multi_to_binary_transform)
        
          # #print multiclass classification evaluation metrics
          # print("multiclass classification evaluation metrics : \n")
          # evaluate_prediction_on_test_set(df_pred.label_true, df_pred.label_pred)
        
          #print binary classification evaluation metrics
          print("binary classification evaluation metrics : \n")
          evaluate_prediction_on_test_set(df_pred.binary_true, df_pred.binary_pred)

          # Create a confusion matrix (POS)
          labels = sorted(list(set(df_pred['binary_true']).union(set(df_pred['binary_pred']))))
          cm = confusion_matrix(df_pred['binary_true'], df_pred['binary_pred'], labels=labels)
        
          # Plot confusion matrix with annotations
          plt.figure(figsize=(10, 8))
          sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
          plt.xlabel('Predicted')
          plt.ylabel('Actual')
          plt.title('Confusion Matrix For Binary Classification')
          plt.show()
            
