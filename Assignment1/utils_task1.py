import os
import pandas as pd
import itertools
from pre_process import *

path = os.path.join(os.path.join(os.getcwd(), 'archive'), 'NER_Dataset.csv')

def load_csv():
    return pd.read_csv(path)

def flatten_list(array):
    return list(itertools.chain.from_iterable(array))

def create_data(dataset):
    txts = []
    posi = []
    tgs = []
    for index, row in dataset.iterrows():
        new_word,new_pos,new_ner = preprocess_pos_ner(row)
        if new_word is not None:
            txts.append(new_word)
            posi.append(new_pos)
            tgs.append(new_ner)
    return txts, posi, tgs

def find_mismatch(true_label, predicted_label):
    mismatch = []
    for idx, (true, predicted) in enumerate(zip(true_label, predicted_label)):
        if len(true)!=len(predicted):
            mismatch.append(idx)
    return mismatch

def match_length(true, predicted):
    mismatch = find_mismatch(true, predicted)
    for idx in mismatch:
        true_length = len(true[idx])
        predicted[idx] = predicted[idx][0:true_length]
    return true, predicted
