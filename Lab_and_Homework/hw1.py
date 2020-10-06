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

log("ok\n")
log("\t{} messages for training\n".format(trainingSize))
log("\t{} messages for testing\n".format(testSize))

##Making the dictionary
log("\nMaking dictionary... ")

listChar="[\'!\"$%&'()*+,-./:;<=>?@[\\]^_|~¡£¥¦¨©¬º»¼¾âãéœˆ˜–‘’“”‰€™],"
def cleanMsg(msg):
    msg = msg.lower()
    for c in listChar:
        msg = msg.replace(c," ")
    return msg

rawDictionary=[]

for message in trainingSet:
    content = cleanMsg(message[1])
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
        content = cleanMsg(message[1])
        words=content.split(" ")
        for word in words:
            try:
                wordIndexInDict=dictionary.index(word)
                features_matrix[i,wordIndexInDict] = 1#words.count(word)
            except: pass #word not in dictionary

    return features_matrix

log("\nExtracting features :\n")
log("\tTraining set... ")
trainingFeaturesMatrix=extractFeaturesMatrix(trainingSet)
log("ok\n")
log("\tTest set... ")
testFeaturesMatrix=extractFeaturesMatrix(testSet)
log("ok\n")

##Fitting Naives Bayes
log("\nFitting Naives Bayes... ")
countWordsInSpam=np.zeros(nbWords)
countWordsInHam=np.zeros(nbWords)
counterSpam=0


for type, message in trainingSet:
    content = cleanMsg(message)
    words=content.split(" ")
    words=np.unique(np.asarray(words))

    if type=="0":
        for word in words:
            try:
                wordIndexInDict=dictionary.index(word)
                countWordsInHam[wordIndexInDict]+=1
            except: pass #word not in dictionary

    elif type=="1":
        counterSpam+=1

        for word in words:
            try:
                wordIndexInDict=dictionary.index(word)
                countWordsInSpam[wordIndexInDict]+=1
            except: pass #word not in dictionary

countWordsInSpam = countWordsInSpam / counterSpam
countWordsInHam = countWordsInHam / (trainingSize - counterSpam)
phi=counterSpam/trainingSize
log("ok\n")

##Testing
log("\nTesting... ")
alpha=0.000000000000000001
predictList=[]

for type, message in testSet:
    content = cleanMsg(message)
    words=content.split(" ")
    words=np.unique(np.asarray(words))
    sumInSpam=0
    sumInHam=0

    for word in words:
        try:
            wordIndexInDict=dictionary.index(word)
            sumInSpam *= countWordsInSpam[wordIndexInDict]
            sumInHam *= countWordsInHam[wordIndexInDict]
        except: pass #word not in dictionary

    p = (sumInSpam * phi + alpha) / (sumInHam * (1-phi) + sumInSpam * phi + 2*alpha)

    predictList.append(p)

predictList=np.asarray(predictList)
log("ok\n")

#Measuring performance
log("\nMeasuring performance... ")

def positive_negative(predict,test):
    true_positive = 0 #predicted spam and is spam
    false_positive = 0 #predicted spam and is not spam
    true_negative = 0 # predicted not spam and is not spam
    false_negative = 0 # predicted not spam and is spam
    N = len(predict)
    for i in range(N):
        if (predict[i] and test[i][0] == "1"):
            true_positive+=1
        elif (not predict[i] and test[i][0] == "0"):
            true_negative+=1
        elif (predict[i] and test[i][0] == "0"):
            false_positive+=1
        elif (not predict[i] and test[i][0] == "1"):
            false_negative+=1
    return true_positive, false_positive, true_negative, false_negative

true_positive, false_positive, true_negative, false_negative = positive_negative(predictList>0.5,testSet)
log("ok\n")

print()
print("           \tSpam  \t  Ham")
print("Pred Spam  \t {}   \t   {}".format(true_positive,false_positive))
print("Pred Ham   \t {}   \t   {}".format(false_negative,true_negative))















