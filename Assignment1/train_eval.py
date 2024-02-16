import pandas as pd
from config import data_path

## Read Data
df = pd.read_csv(data_path)
df = df[df.reviewText.notnull()]
X = df['reviewText']
y = df['overall'].astype(int)

#####################################################################
########## Data Preprocessing steps needs to be added ###############
#####################################################################

# Perform test train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Assuming you have X_test and y_test as the test data and labels
# Define the parameters for each model
dt_params = ['criterion__entropy','criterion__gini']
rf_params = ['n_estimators__20', 'n_estimators__50', 'n_estimators__100']

# Define a list of classifiers with their respective parameters
classifiers = [
    # {'model_name': 'naive_bayes', 'params': nb_params},
    {'model_name': 'decision_tree', 'params': dt_params},
    {'model_name': 'random_forest', 'params': rf_params}
]

vectorizer = "delta_tf_idf"

# Iterate over each classifier and its parameters
for classifier in classifiers:
    model_name = classifier['model_name']
    params = classifier['params']

    for param in params :

        key,val = param.split("__")
        if val.isnumeric() :
          val = int(val)

        param_dict = {key : val}
        # Instantiate the SentimentClassifier object
        clf = SentimentClassifier(model_name,vectorizer, **param_dict)

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
