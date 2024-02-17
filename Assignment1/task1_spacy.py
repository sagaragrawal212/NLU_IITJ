from NLU_IITJ.Assignment1.utils_task1 import *
from NLU_IITJ.Assignment1.evaluation_task1 import *
from tqdm import tqdm
import spacy


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


if __name__ == "__main__":
    texts, posi, tgs = create_data(load_csv())
    predicted_pos_tags, predicted_entities = predict_tag_ner(texts)
    posi, predicted_pos_tags = match_length(posi, predicted_pos_tags)
    tgs, predicted_entities = match_length(tgs, predicted_entities)
    accuracy_pos, confusion_matrix_pos, accuracy_ner, confusion_matrix_ner = accuracy_confusion_matrix(posi, tgs,
                                                                                                       predicted_pos_tags,
                                                                                                       predicted_entities
                                                                                                       )
    print("Accuracy POS tagging:", accuracy_pos)
    print("Confusion Matrix POS tagging:\n", confusion_matrix_pos)
    print("Accuracy NER:", accuracy_ner)
    print("Confusion Matrix NER:\n", confusion_matrix_ner)
