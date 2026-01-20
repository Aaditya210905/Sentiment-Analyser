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
    
    # Convert to sequences
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    # Pad sequences
    X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding=padding, truncating=truncating)
    X_test_padded = pad_sequences(X_test_seq, maxlen=max_length, padding=padding, truncating=truncating)
    
    return X_train_padded, X_test_padded, tokenizer

def preprocess_sample(sample_text, tokenizer, max_length=300, padding='pre', truncating='post'):
    # sample_text must be a Pandas Series
    sample_text = sample_text.apply(prepare_text)
    seq = tokenizer.texts_to_sequences(sample_text)
    padded = pad_sequences(seq, maxlen=max_length, padding=padding, truncating=truncating)
    return padded
