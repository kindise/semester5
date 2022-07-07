# import libraries 
import re
import pickle
import pandas as pd
from tqdm import tqdm
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# define global stopwords
stemmer  = StemmerFactory().create_stemmer()
stopword = set(stopwords.words('indonesian'))

def generate_wordcloud(data):
    comment_words = ''
    for comment in data:
        # apply tokenize and lowercase
        comment_words += " ".join([i.lower() for i in comment.split()]) + " "
        
    wordcloud = WordCloud(width = 800, height = 200,
                background_color ='white',
                stopwords = stopword,
                min_font_size = 10).generate(comment_words)
 
    # plot the wordCloud image                      
    fig = plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)

    return fig

# define cleaning function
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
    # hapus huruf terkait shopee dan kecilkan huruf
    text = " ".join([i.lower() for i in text.split() if i.lower().strip() not in ["shopee", "shopeefood", "shopeefod_id", "food"]])

    return text

casefolding      = lambda x: x.lower()
tokenizing       = lambda x: x.split()
remove_stopword  = lambda x: " ".join([word for word in x if word not in stopword])
single_stem_text = lambda x: stemmer.stem(x)
stem_text        = lambda x: [stemmer.stem(text) for text in tqdm(x, desc = "stemming start...")]
reduce           = lambda x: " ".join([i for i in x.split() if len(i) >= 3])


def preprocess_dataset(df):
    # convert list to series
    df = pd.Series(df)
    
    # proses cleansing
    _cleansing = df.apply(lambda x: cleansing(x))
    print("Cleaning Done...")

    # proses casefolding
    _casefolding = [casefolding(x) for x in _cleansing]
    print("Casefolding Done...")

    # proses tokenizing
    _tokenize = [tokenizing(x) for x in _casefolding]
    print("Tokenizing Done...")

    # proses hapus stopword
    _stopword = [remove_stopword(x) for x in _tokenize]
    print("Stopwords Done...")

    # proses stemming
    _stemming = stem_text(_stopword)
    print("Stemming Done...")

    # proses text reduction
    _reduce = [reduce(x) for x in _stemming]

    return _reduce

def preprocess_only(df):
    # proses cleansing
    _cleansing = df.apply(lambda x: cleansing(x))
    print("Cleaning Done...")

    # proses casefolding
    _casefolding = [casefolding(x) for x in _cleansing]
    print("Casefolding Done...")

    # proses tokenizing
    _tokenize = [tokenizing(x) for x in _casefolding]
    print("Tokenizing Done...")

    # proses hapus stopword
    _stopword = [remove_stopword(x) for x in _tokenize]
    print("Stopwords Done...")

    # proses stemming
    _stemming = stem_text(_stopword)
    print("Stemming Done...")

    # proses text reduction
    _reduce = [reduce(x) for x in _stemming]

    # wrap dataset
    data = pd.DataFrame({
        "cleaning"    : _cleansing,
        "casefolding" : _casefolding, 
        "tokenize"    : _tokenize, 
        "stopwords"   : _stopword,
        "stemming"    : _stemming, 
        "reduce text" : _reduce
    })

    return data

def preprocess_text(text):
    # proses cleansing
    cleaned_test = single_stem_text(remove_stopword(tokenizing(casefolding(cleansing(text)))))

    return cleaned_test

@st.cache(ttl = 12500, allow_output_mutation=True)
def load_model():
    model = pickle.load(open("static/model_naiveBayes.pkl", "rb"))
    return model

@st.cache(ttl = 12500, allow_output_mutation=True)
def load_embedding():
    transformers  = TfidfVectorizer()
    vectorizers   = CountVectorizer(decode_error = "replace", vocabulary = pickle.load(open('static/feature.pkl', 'rb'))) 
    # result_vector = transformers.fit_transform(vectorizers.fit_transform([text])).toarray()
    return transformers, vectorizers