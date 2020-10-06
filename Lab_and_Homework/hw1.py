import os
import numpy as np

def log(msg): print(msg,end="")

filename = 'messages.txt'
trainingProportion = 0.7 #70%

##Importing messages
log("Reading file... ")
messages = np.loadtxt(filename, dtype=str, delimiter='\t')
N = len(messages)
log("ok\n")
log("\t{} messages\n".format(N))

##Binary change

# 1 if spam, 0 if not spam

for value in messages:
    if (value[0]=="spam"):
        value[0]=1
    else:
        value[0]=0

##Splitting train and test data
log("\nSplitting data... ")
trainingSize = int(N * trainingProportion)
testSize = N - trainingSize

trainingSet=messages[:trainingSize]
testSet=messages[trainingSize:]

#Splitting ham and spam
spamTrainingSetIndexes=[(mail[0]=="spam")*1 for mail in trainingSet]

hamTrainingSetIndexes=[ (1-e) for e in spamTrainingSetIndexes]
spamTestSetIndexes=[(mail[0]=="spam")*1 for mail in testSet]
hamTestSetIndexes=[ (1-e) for e in spamTestSetIndexes]

log("ok\n")
log("\t{} messages for training\n".format(trainingSize))
log("\t{} messages for testing\n".format(testSize))

##Making the dictionary
log("\nMaking dictionary... ")
rawDictionary=[]
listChar=",.;!?'"

for message in trainingSet:
    content = message[1].lower()

    for c in listChar:
        content = content.replace(c," ")

    rawDictionary += content.split(" ")

tmp=len(rawDictionary)
rawDictionary=np.asarray(rawDictionary)
rawDictionary=np.unique(rawDictionary)

#Cleaning the dictionary
dictionary=[]
for word in rawDictionary:#We iterate on a copy of the dictionary
    if len(word) > 1 and word.isalpha():
        dictionary.append(word)

nbWords=len(dictionary)
log("ok\n")
log("\t{} unique words found ({} before cleaning)\n".format(nbWords,tmp))

##Extracting features
def extractFeaturesMatrix(set):
    features_matrix=np.zeros((len(set), nbWords))

    for i, message in enumerate(set):
        content = message[1]
        for word in content.split(" "):
            try:
                wordIndexInDict=dictionary.index(word.lower())
                features_matrix[i,wordIndexInDict] += 1
            except: pass #word not in dictionary

log("\nExtracting features :\n")
log("\tTraining set... ")
trainingFeaturesMatrix=extractFeaturesMatrix(trainingSet)
log("ok\n")
log("\tTest set... ")
testFeaturesMatrix=extractFeaturesMatrix(testSet)
log("ok\n")

##Fitting Naives Bayes
log("\nFitting Naives Bayes... ")
log("not done yet\n")

##Testing
log("\nTesting... ")
log("not done yet\n")

##Measuring performance
log("\nMeasuring performance... ")
log("not done yet\n")


