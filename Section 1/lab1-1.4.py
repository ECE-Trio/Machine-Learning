# -*- coding: utf-8 -*-
#Created on Fri Jan 27 22:53:50 2017
#@author: Abhijeet Singh
import os
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

def make_Dictionary(train_dir):
    emails = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    all_words = []

    for mail in emails:#pour chaque fichier contenant un mail
        with open(mail) as m:
            for i, line in enumerate(m):#pour chaque ligne du fichier
                if i == 2:#si c'est le contenu du message, pas le sujet
                    words = line.split()
                    all_words += words

    dictionary = Counter(all_words)

    for item in list(dictionary):#on supprime les mots pas valides
        if item.isalpha() == False:
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]

    dictionary = dictionary.most_common(3000)
    return dictionary

def extract_features(mail_dir):
    files = [os.path.join(mail_dir, fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files), 3000))
    docID = 0

    for fil in files:
        with open(fil) as fi:
            for i, line in enumerate(fi):
                if i == 2:
                    words = line.split()
                    for word in words:
                        wordID = 0
                        for j, d in enumerate(dictionary):
                            if d[0] == word:
                                wordID = j
                                features_matrix[docID, wordID] = words.count(word)
            docID = docID + 1
    return features_matrix

# Create a dictionary of words with its frequency
train_dir = 'ling-spam\\train-mails'
dictionary = make_Dictionary(train_dir)

# Prepare feature vectors per training mail and its labels
train_labels = np.zeros(702) # y=0, ham
train_labels[351:701] = 1 # y=1, spam
train_matrix = extract_features(train_dir)

# Training Naive bayes classifier and its variants
model = MultinomialNB()
model.fit(train_matrix, train_labels)

# Test the unseen mails for Spam
test_dir = 'ling-spam\\test-mails'
test_matrix = extract_features(test_dir)
test_labels = np.zeros(260)
test_labels[130:260] = 1

result = model.predict(test_matrix)

print(confusion_matrix(test_labels, result))