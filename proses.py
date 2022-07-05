import re
import pandas as pd
import numpy as np
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from nltk.corpus import stopwords

def cleansing(text):
    # senang emoticon
    emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', ':d', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
    ])
 
    # sedih emoticon
    emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
    ])
    # menghapus emoticon
    emoticons = emoticons_happy.union(emoticons_sad)
    text = ' '.join([word for word in text.split() if word not in emoticons])
    # hapus mentions
    text = re.sub('@[^\s]+','',text)
    # hapus hashtags
    text = re.sub("#[A-Za-z0-9_]+","", text)
    # hapus url / links
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www.\S+", "", text)
    # hapus tanda baca
    text = re.sub('[^a-zA-Z0-9\s]', '', text)
    # hapus multiple spaces
    text = re.sub('\s+', ' ', text)
    # hapus retweets
    text = re.sub(r'RT', '', text)
    # hapus angka
    text = re.sub(r'[0-9]+', ' ', text)

    return text

def casefolding(text):
    # mengubah karakter menjadi huruf kecil
    text = text.lower()
    return text

def tokenizing(text):
    text = text.split()
    return text

def remove_stopword(text):
    stp = stopwords.words('indonesian')
    text = ' '.join([word for word in text if word not in stp])
    return text

def stem_text(text):
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text
    
stemmer = StemmerFactory().create_stemmer()
# membaca file csv
df = pd.read_csv(filepath_or_buffer='dataset.csv', sep=',', header=0)
# proses cleansing
df['cleansing'] = df['text'].apply(lambda x: cleansing(x))
# proses casefolding
df['casefolding'] = df['cleansing'].apply(lambda x: casefolding(x))
# proses tokenizing
df['tokenizing'] = df['casefolding'].apply(lambda x: tokenizing(x))
# proses hapus stopword
df['stopword'] = df['tokenizing'].apply(lambda x: remove_stopword(x))
# proses stemming
df['stem'] = df['stopword'].apply(lambda x: stem_text(x))
