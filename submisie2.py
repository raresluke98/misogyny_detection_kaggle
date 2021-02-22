import random

import nltk
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB


def tokenize(text):
    '''Generic wrapper around different tokenization methods.
    '''
    return nltk.WordPunctTokenizer().tokenize(text)


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


def get_corpus_vocabulary(corpus):
    '''Write a function to return all the words in a corpus
    '''
    counter = Counter()
    for text in corpus:
        tokens = tokenize(text)
        counter.update(tokens)
    return counter


def text_to_bow(text, wd2idx):
    '''Convert a text to a bag of words representation.
           @  che  .   ,   di  e
    text   0   1   0   2   0   1
    '''
    features = np.zeros(len(wd2idx))
    for token in tokenize(text):
        if token in wd2idx:
            features[wd2idx[token]] += 1
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
        all_features.append(text_to_bow(text, wd2idx))
    all_features = np.array(all_features)
    return all_features


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


def acc(y_true, y_pred):
    return np.mean((y_pred == y_true)) * 100


def f1(y_true, y_pred):
    tp = 0.0
    fp = 0.0
    fn = 0.0
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


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
corpus = train_df['text']
toate_cuvintele = get_corpus_vocabulary(corpus)
wd2idx, idx2wd = get_representation(toate_cuvintele, 100)

data = corpus_to_bow(corpus, wd2idx)
labels = train_df['label'].values

test_data = corpus_to_bow(test_df['text'], wd2idx)
print(test_data.shape)

from sklearn.neighbors import KNeighborsClassifier

train, valid, y_train, y_valid = split(data, labels, 0.3)
print(train.shape)
print(valid.shape)
print(y_train.shape)
print(y_valid.shape)

print(labels)

from timeit import default_timer as timer

''' KNN cu split random'''
start = timer()
v_n, v_a, v_f = [], [], []
for n_n in range(5, 71, 5):
    clf = KNeighborsClassifier(n_neighbors=n_n, n_jobs=-1)
    clf.fit(train, y_train)

    predictii = clf.predict(valid)
    val = acc(y_valid, predictii)
    val1 = f1(y_valid, predictii)

    print("n_neighbours=" + str(n_n) + " acc=" + str(val) + " f1=" + str(val1))
    v_n.append(n_n)
    v_a.append(val)
    v_f.append(val1)

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.suptitle('Acuratete si f1 pentru n vecini')
ax1.plot(v_n, v_a)
ax2.plot(v_n, v_f)
fig.show()
print("timp: ", timer() - start, 's')


