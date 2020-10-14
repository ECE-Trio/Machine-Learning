import numpy as np
from sklearn.naive_bayes import GaussianNB

X = np.array([[-1, -1],[-2, -1],[-3, -2],[1, 1],[2, 1],[3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])

clf = GaussianNB()
clf.fit(X, y)

print(clf.predict([[-1, -0.8]]))


clf_pf = GaussianNB()
clf_pf.partial_fit(X, y, np.unique(y))

print(clf_pf.predict([[-1, -0.8]]))