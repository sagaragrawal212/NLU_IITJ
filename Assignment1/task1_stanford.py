# from NLU_IITJ.Assignment1.evaluation_task1 import *
from evaluation_task1 import *
from pre_process import *
from config import *
import pandas as pd
import nltk
nltk.download('punkt')
from nltk.tag import StanfordNERTagger
from nltk.tag.stanford import StanfordPOSTagger

## downloaded stanford POS tagger and NER using below command  sepratly
# !wget https://nlp.stanford.edu/software/stanford-tagger-4.2.0.zip
# !wget https://nlp.stanford.edu/software/stanford-ner-4.2.0.zip


#unzip both taggers, using belwo commands sepratly
# !unzip -o stanford-tagger-4.2.0.zip
# !unzip -o stanford-ner-4.2.0.zip

# create POS tagger object
pos_tagger = StanfordPOSTagger(model_filename="/content/stanford-postagger-full-2020-11-17/models/english-bidirectional-distsim.tagger",
                           path_to_jar="/content/stanford-postagger-full-2020-11-17/stanford-postagger.jar", encoding='utf-8')
#create NER tagger object
ner_tagger = StanfordNERTagger(model_filename="/content/stanford-ner-2020-11-17/classifiers/english.all.3class.distsim.crf.ser.gz",
                           path_to_jar="/content/stanford-ner-2020-11-17/stanford-ner.jar", encoding='utf-8')

#this function take list tokenize sentences and return POS tags
def predict_pos_tag(list_of_sent):
  # run pos tagger on complete batch in one go
  result_list_pos = pos_tagger.tag_sents(list_of_sent)
  #return only POS tages, not tuple of word and tag
  result_list_pos = [[tup[-1] for tup in each] for each in result_list_pos]
  return result_list_pos

#this function take list tokenize sentences and return NER tags
def predict_ner_tag(list_of_sent):
  result_list_ner = []
  # run pos tagger on batches of 10,000 tokenize sentences
  for i in range(0,len(list_of_sent), 10000):
    res = ner_tagger.tag_sents(list_of_sent[i:i+10000])
    result_list_ner.extend(res)
  #return only POS tages, not tuple of word and tag
  result_list_ner = [[tup[-1] for tup in each] for each in result_list_ner]
  return result_list_ner

if __name__ == "__main__":
  #load csv , #currently loading top 10000 rows for test
  df = pd.read_csv(ner_pos_data_path, nrows=100)
  print("dataset loaded, size : ", df.shape)
  #preprocess data
  df[['new_word', 'new_pos', 'new_ner']] = df.apply(lambda x : preprocess_pos_ner(x), axis = 1,result_type='expand')
 
  #do sanity check
  df['len_new_word'] = df.new_word.apply(lambda x : len(x) if x != None else 0)
  df['len_new_pos'] = df.new_pos.apply(lambda x : len(x) if x != None else 0)
  df['len_new_ner'] = df.new_ner.apply(lambda x : len(x) if x != None else 0)
  if (df['len_new_word'] == df['len_new_pos']).all():
    print("sanity check done for pos")
  if (df['len_new_word'] == df['len_new_ner']).all():
    print("sanity check done for ner")
 
  #drop None 
  df.dropna(inplace = True)
  print("dataset ready, size : ", df.shape)
  #predict POS using stanford tagger
  print("running pos tag prediction...")
  df['pos_pred'] = predict_pos_tag(df.new_word)
  
  #predict NER using stanford tagger
  print("running ner tag prediction...")
  df['ner_pred']  = predict_ner_tag(df.new_word)

  ## add stanford tags to dataset tags mapping
  #NO requirement in cas of POS tagging

  #add mapping dict for NER
  ner_dataset_to_stanford = {
            'B-art' : 'O',
            'B-eve' : 'O',
            'B-geo' : 'LOCATION',
            'B-gpe' : 'LOCATION',
            'B-nat':  'O',
            'B-org' : 'ORGANIZATION',
            'B-per' : 'PERSON',
            'B-tim' : 'O',
            'I-art' : 'O',
            'I-eve' : 'O',
            'I-geo' :  'LOCATION',
            'I-gpe': 'LOCATION',
            'I-nat': 'O',
            'I-org': 'ORGANIZATION',
            'I-per' : 'PERSON',
            'I-tim' : 'O',
            'O' : 'O'
            }
  df.new_ner_mapped = df.new_ner.apply(lambda x : [ner_dataset_to_stanford[each] for each in x])

  #evaluation
  accuracy_pos,cm_pos,accuracy_ner,cm_ner = accuracy_confusion_matrix(df.new_pos.tolist(),
                                                                  df.new_ner_mapped.tolist(), #df.new_ner.tolist(),
                                                                  df.pos_pred.tolist(),
                                                                  df.ner_pred.tolist() )
  print("Accuracy POS tagging:", accuracy_pos)
  
  import itertools
  def flatten_list(array):
      return list(itertools.chain.from_iterable(array))
  
  print("Confusion Matrix POS tagging:\n", cm_pos)
  df_res_pos = pd.DataFrame({"actual" : flatten_list(df.new_pos.tolist()),
              "predicted" : flatten_list(df.pos_pred.tolist())})
  
  df_res_ner = pd.DataFrame({"actual" : flatten_list(df.new_ner_mapped.tolist()),
              "predicted" : flatten_list(df.ner_pred.tolist())})
  df_res_ner['predicted'] = df_res_ner.predicted.apply(lambda text : text.replace("\n",''))
  
  # Create a confusion matrix (POS)
  labels = sorted(list(set(df_res_pos['actual']).union(set(df_res_pos['predicted']))))
  cm = confusion_matrix(df_res_pos['actual'], df_res_pos['predicted'], labels=labels)
  
  # Plot confusion matrix with annotations
  plt.figure(figsize=(10, 8))
  sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  plt.title('Confusion Matrix POS')
  plt.show()
  
  print("Accuracy NER:", accuracy_ner)
    # Create a confusion matrix (NER)
  labels = sorted(list(set(df_res_ner['actual']).union(set(df_res_ner['predicted']))))
  cm_ner = confusion_matrix(df_res_ner['actual'], df_res_ner['predicted'], labels=labels)
  print("Confusion Matrix NER:\n", cm_ner)
  # Plot confusion matrix with annotations
  plt.figure(figsize=(10, 8))
  sns.heatmap(cm_ner, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  plt.title('Confusion Matrix NER')
  plt.show()

  # #evaluation
  # accuracy_pos,cm_pos,accuracy_ner,cm_ner = accuracy_confusion_matrix(df.new_pos.tolist(),
  #                                                                   df.new_ner_mapped.tolist(), #df.new_ner.tolist(),
  #                                                                   df.pos_pred.tolist(),
  #                                                                   df.ner_pred.tolist() )
  # print("Accuracy POS tagging:", accuracy_pos)
  # # print("Confusion Matrix POS tagging:\n", cm_pos)
  # print("Accuracy NER:", accuracy_ner)
  # # print("Confusion Matrix NER:\n", cm_ner)
  # print("CM size", len(cm_pos), len(cm_ner))
