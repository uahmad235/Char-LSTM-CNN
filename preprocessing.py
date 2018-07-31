import pandas as pd
import numpy as np



class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


def load_dataset(path_to_dataset = "C:\\Users\\Usman Ahmad\\Downloads\\Compressed\\entity-annotated-corpus\\ner_dataset.csv"):

	data = pd.read_csv(path_to_dataset, encoding="latin1") 

	data = data.fillna(method="ffill")

	# prints last 10 rows of data
	print(data.tail(10))

	#extract words from dataset
	words = list(set(data["Word"].values))

    # words.append("UNK")
	# words.append("ENDPAD")
    
	# extract tags from words
	tags = list(set(data["Tag"].values))
	pos = list(set(data["POS"].values))

	return data , words, tags, pos

def word2features(sent, i):

    word = sent[i][0]
    postag = sent[i][1]

    # feature of curernt word
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }

    # features for prev word, if not first word of sentence
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    # features for next word, if not last word of sentence
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    # if word is last in a sentence, add  "End Of Sentence" feature
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
	""" returns a "list of features" for "list of words" in sentence """
	return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
	""" returns list of label against specific token """
	return [label for token, postag, label in sent]


def get_X_y(data):
	""" returns the data in x and y"""
	getter = SentenceGetter(data)

	# list of (word,POS,Tag)
	sentences = getter.sentences

	# features_sent = sent2features(sentences)
	# print(features_sent[:5])

	X = [sent2features(s) for s in sentences]
	y = [sent2labels(s) for s in sentences]

	return X, y
