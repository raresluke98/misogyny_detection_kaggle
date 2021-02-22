from collections import Counter
import random
import math
import inline as inline
import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import string
import re
# from nltk.stem.porter import *
# %matplotlib inline
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from timeit import default_timer as timer

pd.set_option('display.max_colwidth', 100)


def load_data():
    data = pd.read_csv('train.csv')
    data_test = pd.read_csv('test.csv')
    return data, data_test


tweet_df, tweet_df_test = load_data()
print(tweet_df.head())

print('Dataset size:', tweet_df.shape)
print('Columns are:', tweet_df.columns)

tweet_df.info()

sns.countplot(x='label', data=tweet_df)
# plt.show()

df = pd.DataFrame(tweet_df[['id', 'text']])
df_test = pd.DataFrame(tweet_df_test[['id', 'text']])

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# Start w/ one review
df_1 = tweet_df[tweet_df['label'] == 1]
df_0 = tweet_df[tweet_df['label'] == 0]
tweet_All = " ".join(review for review in df.text)
tweet_1 = " ".join(review for review in df_1.text)
tweet_0 = " ".join(review for review in df_0.text)

fig, ax = plt.subplots(3, 1, figsize=(30, 30))

# Create and generate a word cloud image
wordcloud_ALL = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(tweet_All)
wordcloud_1 = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(tweet_1)
wordcloud_0 = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(tweet_0)

# Display the generated image:
ax[0].imshow(wordcloud_ALL, interpolation='bilinear')
ax[0].set_title('All Tweets', fontsize=30)
ax[0].axis('off')
ax[1].imshow(wordcloud_1, interpolation='bilinear')
ax[1].set_title('Tweets under 1 class', fontsize=30)
ax[1].axis('off')
ax[2].imshow(wordcloud_0, interpolation='bilinear')
ax[2].set_title('Tweets under 0 class', fontsize=30)
ax[2].axis('off')

plt.show()


# High freq tokens in both classes: https, che, e, t

# Pre-processing text data

# Remove punctuations

def remove_punct(text):
    text = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text


df['text_punct'] = df['text'].apply(lambda x: remove_punct(x))
df_test['text_punct'] = df_test['text'].apply(lambda x: remove_punct(x))
print(df.head(10))


def tokenization(text):
    text = re.split('\W+', text)
    return text


df['text_tokenized'] = df['text_punct'].apply(lambda x: tokenization(x.lower()))
df_test['text_tokenized'] = df_test['text_punct'].apply(lambda x: tokenization(x.lower()))
# df.head()

stopword = nltk.corpus.stopwords.words('italian')
stopword.extend(['https', 'che', 'e', 't'])


def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text


df['text_nonstop'] = df['text_tokenized'].apply(lambda x: remove_stopwords(x))
df_test['text_nonstop'] = df_test['text_tokenized'].apply(lambda x: remove_stopwords(x))
df.head()

ps = nltk.SnowballStemmer("italian")


def stemming(text):
    text = [ps.stem(word) for word in text]
    return text


df['text_stemmed'] = df['text_nonstop'].apply(lambda x: stemming(x))
df_test['text_stemmed'] = df_test['text_nonstop'].apply(lambda x: stemming(x))
df.head()


def get_corpus_vocabulary(corpus):
    '''Write a function to return all the words in a corpus
    '''
    counter = Counter()
    for text in corpus:
        counter.update(text)
    return counter


corpus = df['text_stemmed']
corpus_test = df_test['text_stemmed']
toate_cuvintele = get_corpus_vocabulary(corpus)


# toate_cuvintele_test = get_corpus_vocabulary(corpus_test) ELIMINAT !!!

def get_representation(toate_cuvintele, how_many):
    '''Extract the first most common words from a vocabulary
    and return two dictionaries: word to index and index to word
           @  che  .   ,   di  e
    text0  0   1   0   2   0   1
    text1  1   2 ...
    ...
    textN  0   0   1   1   0   2
    '''
    most_comm = toate_cuvintele.most_common(how_many)
    wd2idx = {}
    idx2wd = {}
    for idx, itr in enumerate(most_comm):
        cuvant = itr[0]
        wd2idx[cuvant] = idx
        idx2wd[idx] = cuvant
    return wd2idx, idx2wd


wd2idx, idx2wd = get_representation(toate_cuvintele, 100)


# wd2idx_test, idx2wd_test = get_representation(toate_cuvintele_test, 100) ELIMINAT !!!

def normalizare_l2(features):
    suma = 0.000001
    for nr in features:
        suma += nr * nr
    rad_suma = math.sqrt(suma)
    return features / rad_suma


def text_to_bow(text, wd2idx):
    '''Convert a text to a bag of words representation.
           @  che  .   ,   di  e
    text   0   1   0   2   0   1
    '''
    features = np.zeros(len(wd2idx))
    for cuvant in text:
        if cuvant in wd2idx:
            features[wd2idx[cuvant]] += 1
    return features


def corpus_to_bow(corpus, wd2idx):
    '''Convert a corpus to a bag of words representation.
               @  che  .   ,   di  e
        text0  0   1   0   2   0   1
        text1  1   2 ...
        ...
        textN  0   0   1   1   0   2
    '''
    all_features = []
    for text in corpus:
        all_features.append(normalizare_l2(text_to_bow(text, wd2idx)))  ####MODIFICARE
    all_features = np.array(all_features)
    return all_features


data = corpus_to_bow(corpus, wd2idx)
data_test = corpus_to_bow(corpus_test, wd2idx)
labels = tweet_df['label'].values


def acc(y_true, y_pred):
    return np.mean((y_pred == y_true)) * 100


def f1(y_true, y_pred):
    tp = 0.000001
    fp = 0.000001
    fn = 0.000001
    for elem_true, elem_pred in zip(y_true, y_pred):
        if elem_pred == 1 and elem_true == 1:
            tp = tp + 1
        if elem_pred == 1 and elem_true == 0:
            fp = fp + 1
        if elem_pred == 0 and elem_true == 1:
            fn = fn + 1
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return (2 * (p * r)) / (p + r)


def elemente_m_confuzie(y_true, y_pred):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for elem_true, elem_pred in zip(y_true, y_pred):
        if elem_pred == 1 and elem_true == 1:
            tp = tp + 1
        if elem_pred == 1 and elem_true == 0:
            fp = fp + 1
        if elem_pred == 0 and elem_true == 1:
            fn = fn + 1
        if elem_pred == 0 and elem_true == 0:
            tn = tn + 1
    return tp, fp, fn, tn


def write_prediction(out_file, predictions):
    '''A function to write the predictions to a file.
    id,label
    5001,1
    5002,1
    5003,1
    ...
    '''
    with open(out_file, 'w') as fout:
        # aici e open in variabila 'fout'
        fout.write('id,label\n')
        start_id = 5001
        for i, pred in enumerate(predictions):
            linie = str(i + start_id) + ',' + str(pred) + '\n'
            fout.write(linie)
    # aici e fisierul closed


def split(data, labels, procentaj_valid=0.25):
    '''75% train, 25% valid
    important! mai intai facem shuffle la date
    '''
    indici = np.arange(0, len(labels))
    random.shuffle(indici)
    N = int((1 - procentaj_valid) * len(labels))
    train = data[indici[:N]]
    valid = data[indici[N:]]
    y_train = labels[indici[:N]]
    y_valid = labels[indici[N:]]
    return train, valid, y_train, y_valid


def cross_validate(k, data, labels):
    '''Split the data into k chunks.
    iteration 0:
        chunk 0 is for validation, chunk[1:] for train
    iteration 1:
        chunk 1 is for validation, chunk[0] + chunk[2:] for train
    ...
    iteration k:
        chunk k is for validation, chunk[:k] for train
    '''
    chunk_size = len(labels) // k
    indici = np.arange(0, len(labels))
    random.shuffle(indici)
    for i in range(0, len(labels), chunk_size):
        valid_indici = indici[i:i + chunk_size]
        train_indici = np.concatenate([indici[0:i], indici[i + chunk_size:]])
        train = data[train_indici]
        valid = data[valid_indici]
        y_train = labels[train_indici]
        y_valid = labels[valid_indici]
        yield train, valid, y_train, y_valid


'''Evaluare mnb utilizand cross_validate
'''
print('EVALUARE MODEL MNB ################')
start = timer()
for alfa in [0.000001, 0.2, 0.4, 0.6, 0.8, 1]:
    mnb = MultinomialNB(alpha=alfa)
    v_f = []
    for idx in range(1000):
        for train, valid, y_train, y_valid in cross_validate(10, data, labels):
            mnb.fit(train, y_train)
            y_pred = mnb.predict(valid)
            val = f1(y_valid, y_pred)
            # print('f1: ',val)
            v_f.append(val)
    print('alpha: ', alfa, 'f1: ', np.mean(v_f))
print('timp: ', timer() - start, 's  egal cu:', (timer() - start) / 60, 'min')

print()
print('10 fold cv si matrice de confuzie')

tp, fp, fn, tn = 0, 0, 0, 0
v_a = []
v_f = []
for train, valid, y_train, y_valid in cross_validate(10, data, labels):
    mnb.fit(train, y_train)
    y_pred = mnb.predict(valid)
    val = acc(y_valid, y_pred)
    val1 = f1(y_valid, y_pred)
    tp1, fp1, fn1, tn1 = elemente_m_confuzie(y_valid, y_pred)
    tp += tp1
    fp += fp1
    fn += fn1
    tn += tn1
    v_a.append(val)
    v_f.append(val1)

print('acc:', np.mean(v_a), 'f1: ', np.mean(v_f))

print('Matricea de confuzie:')

print(tp, ' ', fp, '\n',
      fn, ' ', tn)

print()

'''Scriere in fisier MNB
'''
# mnb = MultinomialNB(alpha=1)
# predictii = mnb.fit(data, labels).predict(data_test)
# write_prediction('predictii_full_cleaning_MNB_alpha_10.csv', predictii)

print('fin')
