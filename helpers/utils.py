import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

def filter_text_by_words(data, n=10):
    return data[[len(re.findall(r'\w+', x)) > n for x in data.text]]


stop_words = set(stopwords.words('english'))

class LemmaTokenizer:
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`']
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        # Remove all the special characters
        doc = re.sub(r'\W', ' ', doc)
        # remove all single characters
        doc = re.sub(r'\s+[a-zA-Z]\s+', ' ', doc)
        # Remove single characters from the start
        doc = re.sub(r'\^[a-zA-Z]\s+', ' ', doc)
        # Substituting multiple spaces with single space
        doc = re.sub(r'\s+', ' ', doc, flags=re.I)

        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t not in self.ignore_tokens]

def return_tokenizer():
    tokenizer=LemmaTokenizer()
    token_stop = tokenizer(' '.join(stop_words) + 'www http com')
    return token_stop, tokenizer

def get_vectorizer():
    token_stop, tokenizer = return_tokenizer()
    vectorizer = TfidfVectorizer(lowercase=True, stop_words=token_stop, ngram_range=(1,2), max_df=0.7, min_df=5, tokenizer=tokenizer)
    return vectorizer

def combine_lists(x):
    res = set()
    for ix in x:
        res = res.union(ix)
    return res