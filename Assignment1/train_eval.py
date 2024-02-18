import pandas as pd
import numpy as np
from config import data_path,vectorizer,feature_selection
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from SentimentClassifier import SentimentClassifier
from pre_process import preprocess_text_task2

## Read Data
print("Loading Data ...")
df = pd.read_csv(data_path, nrows = 100)
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
        pred = clf.predict(X_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, pred)
        # precision = precision_score(y_test, pred)
        # recall = recall_score(y_test, pred)
        # f1score = f1_score(y_test, pred)
        confusion = confusion_matrix(y_test, pred)

        # Print the results
        print(f"Results for {model_name} with parameters {param_dict}:")
        print("Accuracy:", accuracy)
        # print("Precision:", precision)
        # print("Recall:", recall)
        # print("F1 Score:", f1score)
        print("Confusion Matrix:\n", confusion)
        print()
