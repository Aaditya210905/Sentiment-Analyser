import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

lemmatizer = WordNetLemmatizer()

def prepare_text(text):

    # Lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Remove emojis
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub('', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Lemmatize
    words = text.split()
    text = ' '.join([lemmatizer.lemmatize(word) for word in words])

    return text

def preprocess(X_train=None, X_test=None,tokenizer=None, num_words=16000, 
               max_length=300, oov_token='<OOV>', padding='pre', truncating='post'):
    
    X_train = X_train.apply(prepare_text)
    X_test = X_test.apply(prepare_text)
    # Fit tokenizer and transform training/test data
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
    tokenizer.fit_on_texts(X_train)
    
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    
    X_train = pad_sequences(X_train, maxlen=max_length, padding=padding, truncating=truncating)
    X_test = pad_sequences(X_test, maxlen=max_length, padding=padding, truncating=truncating)
    
    return X_train, X_test, tokenizer

def preprocess_sample(sample, tokenizer, max_length=300, padding='pre', truncating='post'):
    sample = sample.apply(prepare_text)
    sample = tokenizer.texts_to_sequences(sample)
    sample = pad_sequences(sample, maxlen=max_length, padding=padding, truncating=truncating)
    return sample
