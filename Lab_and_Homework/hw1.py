import os
import numpy as np

filename = 'messages.txt'
trainingProportion = 0.7 #70%

#Importing messages
messages = np.loadtxt(filename, dtype=str, delimiter='\t')

N = len(messages)
trainingSize = int(N * trainingProportion)
testSize = N - trainingSize

#Splitting train and test data
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

#Making the dictionary
dictionary=[]
for message in trainingSet:
    content = message[1]
    dictionary += content.split(" ")

dictionary=np.asarray(dictionary)
dictionary=np.unique(dictionary)













