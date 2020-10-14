import numpy as np
from sklearn.naive_bayes import BernoulliNB

X = np.random.randint(2, size=(6, 100))
y = np.array([1, 2, 3, 4, 4, 5])

clf = BernoulliNB()
clf.fit(X, y)


print(clf.predict(X))

for i in range(0,6):
    #print(clf.predict(X[i:(i+1)]))
    print(clf.predict([X[i]]))