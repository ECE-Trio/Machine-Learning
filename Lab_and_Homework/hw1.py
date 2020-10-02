import os
import numpy as np

def log(msg): print(msg,end="")

filename = 'messages.txt'
trainingProportion = 0.7 #70%

##Importing messages
log("Reading file... ")
messages = np.loadtxt(filename, dtype=str, delimiter='\t')
log("ok\n")

N = len(messages)
trainingSize = int(N * trainingProportion)
testSize = N - trainingSize

##Splitting train and test data
log("Splitting data... ")
trainingSet=messages[:trainingSize]
testSet=messages[trainingSize:]

#Splitting ham and spam
spamTrainingSetIndexes=[(mail[0]=="spam") for mail in trainingSet]
hamTrainingSetIndexes=[not e for e in spamTrainingSetIndexes]
spamTestSetIndexes=[(mail[0]=="spam") for mail in testSet]
hamTestSetIndexes=[not e for e in spamTestSetIndexes]

spamTrainingSet=trainingSet[spamTrainingSetIndexes]
hamTrainingSet=trainingSet[hamTrainingSetIndexes]
spamTestSet=testSet[spamTestSetIndexes]
hamTestSet=testSet[hamTestSetIndexes]

#Dropping first columns
spamTrainingSet=np.delete(spamTrainingSet, 0, axis=1)
hamTrainingSet=np.delete(hamTrainingSet, 0, axis=1)
spamTestSet=np.delete(spamTestSet, 0, axis=1)
hamTestSet=np.delete(hamTestSet, 0, axis=1)
log("ok\n")

##Making the dictionary
log("Making dictionary... ")
rawDictionary=[]
for message in trainingSet:
    content = message[1].lower()
    rawDictionary += content.split(" ")

rawDictionary=np.asarray(rawDictionary)
rawDictionary=np.unique(rawDictionary)

#Cleaning the dictionary
dictionary=[]
for word in rawDictionary:#We iterate on a copy of the dictionary
    if len(word) > 1 and word.isalpha():
        dictionary.append(word)

nbWords=len(dictionary)
log("ok\n")

##Extracting features
log("Extracting features... ")
features_matrix=np.zeros((trainingSize, nbWords))

for i, message in enumerate(trainingSet):
    content = message[1]
    for word in content.split(" "):
        try:
            wordIndexInDict=dictionary.index(word)
            features_matrix[i,wordIndexInDict] += 1
        except: pass #word not in dictionary

log("ok\n")

##Fitting Naives Bayes
log("Fitting Naives Bayes... ")
log("not done yet\n")

##Testing
log("Testing... ")
log("not done yet\n")

##Measuring performance
log("Measuring performance... ")
log("not done yet\n")


