import nltk
from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize
from nltk.corpus import inaugural
import ast
import pandas as pd
from evaluation_task1 import *
from pre_process import *
from config import *
from sklearn.metrics import confusion_matrix
from utils_task1 import flatten_list
from config import ner_pos_data_path
import matplotlib.pyplot as plt
import seaborn as sns
# Download NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')


# Part-of-Speech Tagging (POS)
def pos_tagging(text):
    pos_tags = pos_tag(text)

    result_words = [tup[0] for tup in pos_tags]
    result_pos = [tup[1] for tup in pos_tags]
    return [result_words , result_pos]

def pos_val(pos_gt, pos_pred) :

  return [gt==pred for gt,pred in zip(pos_gt,pos_pred)]


# Named Entity Recognition (NER)
def named_entity_recognition(text):
    pos_tags = pos_tag(text)
    named_entities = ne_chunk(pos_tags)
    ner_op = []
    for ne in named_entities :
      if "/" in str(ne) :
        if len(str(ne).split(" ")) > 2:
           ner_len = len([word.split("/")[0] for word in  str(ne).split(" ") if '/' in word])
           ner = [str(ne).split(" ")[0][1:]]*ner_len
           ner_op.extend(ner)
           continue
        word = str(ne).split(" ")[1].split("/")[0]
        ner = str(ne).split(" ")[0][1:]
        ner_op.append(ner)
      else :
        ner_op.append("O")

    return ner_op

if __name__ == "__main__" :

  df = pd.read_csv(ner_pos_data_path)
  df['Cleaned_Word'], df['Cleaned_POS'], df['Cleaned_NER'] = zip(*df.apply(preprocess_pos_ner, axis=1))
  df = df[df.Cleaned_Word.notnull()]
  df = df.reset_index(drop = True)

  ## POS Tagging
  pos_tag_df = df['Cleaned_Word'].apply(lambda x : pos_tagging(x))
  df["result_words"] = [pos_tag_df[i][0] for i in range(len(pos_tag_df))]
  df["result_pos"] = [pos_tag_df[i][1] for i in range(len(pos_tag_df))]
  # df["pos_validation"] = df.apply(lambda row : pos_val(row['Cleaned_POS'], row['result_pos']),axis = 1)
  # df["pos_flag"] = df.pos_validation.apply(lambda x : sum(x))

  ## NER Tagging 
  df['result_ner'] = df['Cleaned_Word'].apply(lambda x : named_entity_recognition(x))

  #add mapping dict for NER
  ner_dataset_to_nltk = {
                          "B-art": "O",
                          "B-eve": "O",
                          "B-geo": "GPE",
                          "B-gpe": "GPE",
                          "B-nat": "O",
                          "B-org": "ORGANIZATION",
                          "B-per": "PERSON",
                          "B-tim": "O",
                          "I-art": "O",
                          "I-eve": "O",
                          "I-geo": "GPE",
                          "I-gpe": "GPE",
                          "I-nat": "O",
                          "I-org": "ORGANIZATION",
                          "I-per": "PERSON",
                          "I-tim": "O",
                          "O": "O"
                      }
  reverse_map = {"LOCATION" : "GPE","GSP" : "GPE"}
  df['new_ner_mapped'] = df.Cleaned_NER.apply(lambda x : [ner_dataset_to_nltk[each] for each in x])
  df['result_ner'] = df.result_ner.apply(lambda x : [reverse_map[each] if each == "LOCATION" else each for each in x ])

  #evaluation
  accuracy_pos,cm_pos,accuracy_ner,cm_ner = accuracy_confusion_matrix(df.Cleaned_POS.tolist(),
                                                                    df.new_ner_mapped.tolist(), #df.new_ner.tolist(),
                                                                    df.result_pos.tolist(),
                                                                    df.result_ner.tolist() )
  print("Accuracy POS tagging:", accuracy_pos)
  print("Accuracy NER:", accuracy_ner)

  df_res_pos = pd.DataFrame({"actual" : flatten_list(df.Cleaned_POS.tolist()),
              "predicted" : flatten_list(df.result_pos.tolist())})
  
  df_res_ner = pd.DataFrame({"actual" : flatten_list(df.new_ner_mapped.tolist()),
              "predicted" : flatten_list(df.result_ner.tolist())})
  df_res_ner['predicted'] = df_res_ner.predicted.apply(lambda text : text.replace("\n",''))

  # Create a confusion matrix (POS)
  labels = sorted(list(set(df_res_pos['actual']).union(set(df_res_pos['predicted']))))
  cm = confusion_matrix(df_res_pos['actual'], df_res_pos['predicted'], labels=labels)

  # Plot confusion matrix with annotations
  plt.figure(figsize=(20, 16))
  sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  plt.title('Confusion Matrix POS')
  plt.show()


  # Create a confusion matrix (NER)
  labels = sorted(list(set(df_res_ner['actual']).union(set(df_res_ner['predicted']))))
  cm = confusion_matrix(df_res_ner['actual'], df_res_ner['predicted'], labels=labels)

  # Plot confusion matrix with annotations
  plt.figure(figsize=(10, 8))
  sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  plt.title('Confusion Matrix NER')
  plt.show()
