from utils_task1 import *
from evaluation_task1 import *
from tqdm import tqdm
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

new_ner_mapping = {'B-cardinal': 'O',
                   'B-date': 'O',
                   'B-event': 'B-eve',
                   'B-fac': 'O',
                   'B-gpe': 'B-gpe',
                   'B-language': 'O',
                   'B-law': 'O',
                   'B-loc': 'B-geo',
                   'B-money': 'O',
                   'B-norp': 'O',
                   'B-ordinal': 'O',
                   'B-org': 'B-org',
                   'B-percent': 'O',
                   'B-person': 'B-per',
                   'B-product': 'O',
                   'B-quantity': 'O',
                   'B-time': 'B-tim',
                   'B-work_of_art': 'O',
                   'I-cardinal': 'O',
                   'I-date': 'O',
                   'I-event': 'B-eve',
                   'I-fac': 'O',
                   'I-gpe': 'I-gpe',
                   'I-law': 'O',
                   'I-loc': 'I-geo',
                   'I-money': 'O',
                   'I-norp': 'O',
                   'I-org': 'B-org',
                   'I-percent': 'O',
                   'I-person': 'B-per',
                   'I-product': 'O',
                   'I-quantity': 'O',
                   'I-time': 'B-tim',
                   'I-work_of_art': 'O',
                   'O': 'O'
                   }


def predict_tag_ner(texts):
    nlp = spacy.load("en_core_web_trf")
    predicted_pos_tags = []
    predicted_entities = []
    for text in tqdm(texts):
        doc = nlp(" ".join(text))
        predicted_pos_tags.append([token.tag_ for token in doc])
        predicted_entities.append([token.ent_iob_ + '-' + token.ent_type_.lower() if token.ent_iob_ != 'O' else 'O'
                                   for token in doc])
    return predicted_pos_tags, predicted_entities

def apply_mapping(ner_tags, mapping):
  mapped_tags = []
  for sentence_tags in ner_tags:
    mapped_sentence_tags = []
    for tag in sentence_tags:
      mapped_tag = mapping.get(tag, 'O')
      mapped_sentence_tags.append(mapped_tag)
    mapped_tags.append(mapped_sentence_tags)
  return mapped_tags

def plot_confusion_matrix(confusion_matrix, labels, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(20, 20))
    sns.set(font_scale=1.2)
    sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap=cmap, xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    texts, posi, tgs = create_data(load_csv())
    predicted_pos_tags, predicted_entities = predict_tag_ner(texts)
    posi, predicted_pos_tags = match_length(posi, predicted_pos_tags)
    tgs, predicted_entities = match_length(tgs, predicted_entities)

    predicted_tags = apply_mapping(predicted_entities, new_ner_mapping)

    accuracy_pos, confusion_matrix_pos, accuracy_ner, confusion_matrix_ner = accuracy_confusion_matrix(posi, tgs,
                                                                                                       predicted_pos_tags,
                                                                                                       predicted_tags
                                                                                                       )
    print("Accuracy POS tagging:", accuracy_pos)
    print("Confusion Matrix POS tagging:\n", confusion_matrix_pos)
    print("Accuracy NER:", accuracy_ner)
    print("Confusion Matrix NER:\n", confusion_matrix_ner)

    pos_labels = sorted(list(set(flatten_list(posi)).union(set(flatten_list(predicted_pos_tags)))))
    ner_labels = sorted(list(set(flatten_list(tgs)).union(set(flatten_list(predicted_tags)))))

    plot_confusion_matrix(confusion_matrix_pos, labels=pos_labels, title='POS Tag Confusion Matrix')
    plot_confusion_matrix(confusion_matrix_ner, labels=ner_labels, title='NER Tag Confusion Matrix')
