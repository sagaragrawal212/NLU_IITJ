import string
import ast
#imports for preprocessing task 2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
# import string
import re
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def lowercase_text(text):
    # Convert the text to lowercase
    return text.lower()

def remove_punctuation(text):
    # Remove punctuation from the text
    punctuations = """!"#%&'()*+,-/:;<=>?[\]^_`{|}~'""" #except . @ $
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

def preprocess_text_task2(text):
    # Lowercasing
    text = text.lower()

    # Removing URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Removing emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    # Removing special characters, keeping letters numbers,
    # text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'[^\w\s]|_', '', text)

    # Removing punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenization
    tokens = word_tokenize(text)

    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    # Joining tokens back into a single string
    preprocessed_text = ' '.join(lemmatized_tokens)

    return preprocessed_text
