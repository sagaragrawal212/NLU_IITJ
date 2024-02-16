from sklearn.metrics import accuracy_score, confusion_matrix

def accuracy_confusion_matrix(true_tag, true_ner, predicted_tag, predicted_ner):
    accuracy_pos = accuracy_score(flatten_list(true_tag), flatten_list(predicted_tag))
    confusion_matrix_pos = confusion_matrix(flatten_list(true_tag), flatten_list(predicted_tag))
    accuracy_ner = accuracy_score(flatten_list(true_ner), flatten_list(predicted_ner))
    confusion_matrix_ner = confusion_matrix(flatten_list(true_ner), flatten_list(predicted_ner))
    return accuracy_pos, confusion_matrix_pos, accuracy_ner, confusion_matrix_ner