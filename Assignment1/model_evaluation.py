import pandas as pd
# accuracy, Precision, Recall, F1 score and confusion matrix.
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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
  print("Confusion Matrix:\n", conf_matrix)
  return accuracy, precision, recall, f1, conf_matrix

if __name__ == '__main__':
  test_df = pd.read_csv('test_results.csv')
  test_df['binary_true'] = test_df.label_true.apply(multi_to_binary_transform)
  test_df['binary_pred'] = test_df.label_pred.apply(multi_to_binary_transform)

  #print multiclass classification evaluation metrics
  print("multiclass classification evaluation metrics : \n")
  evaluate_prediction_on_test_set(df.label_true, df.label_pred)

  #print binary classification evaluation metrics
  print("binary classification evaluation metrics : \n")
  evaluate_prediction_on_test_set(test_df.binary_true, test_df.binary_pred)
