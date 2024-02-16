import string
import ast

def lowercase_text(text):
    # Convert the text to lowercase
    return text.lower()

def remove_punctuation(text):
    # Remove punctuation from the text
    punctuations = """!"#$%&'()*+,-/:;<=>?[\]^_`{|}~'""" #except . @
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def clean_ascii(word):
    # Keep only alphanumeric characters
    cleaned_word = ''.join(char for char in word if char.isalnum())
    return cleaned_word

def remove_hyphen(word):
    # Remove hyphens
    cleaned_word = word.replace('-', '')
    return cleaned_word

def preprocess_pos_ner(row) :

    text_tokens = ast.literal_eval(row['Word'])
    pos_tokens = ast.literal_eval(row['POS'])
    ner_tokens = ast.literal_eval(row['Tag'])

    clean_sentence = []
    drop_index_list = []

    for index,word in enumerate(text_tokens):
      word_copy = word
      word = clean_ascii(word)
      word = remove_hyphen(word)
      # word = lowercase_text(word)
      word = remove_punctuation(word)
      clean_sentence.append(word)

      if len(word) == 0 :
        drop_index_list.append(index)

    new_word = [value for index, value in enumerate(clean_sentence) if index not in drop_index_list]
    new_pos = [value for index, value in enumerate(pos_tokens) if index not in drop_index_list]
    new_ner = [value for index, value in enumerate(ner_tokens) if index not in drop_index_list]

    if len(new_word) == 0 :
      return None , None , None
    else :
      return new_word,new_pos,new_ner
