"""
Made by the team :
    Olivia Dalmasso
    Alexis Direz
    Neil Ségard

Members³ of : ING5 SI DBA GR1.A
"""
#Import

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture

# Variables
Z = [0,1,2,3]
X = [0,1,2,3,4]
X_target = [1,3,2,0]

a=np.asarray([[1.,  0.,  0.,  0.], #Size of a : NxN
             [0.2, 0.3, 0.1, 0.4],
             [0.2, 0.5, 0.2, 0.1],
             [0.7, 0.1, 0.1, 0.1]])

b=np.asarray([[1., 0.,  0.,  0.,  0.], #Size of b : NxK
             [0., 0.3, 0.4, 0.1, 0.2],
             [0., 0.1, 0.1, 0.7, 0.1],
             [0., 0.5, 0.2, 0.1, 0.2]])

def calculAlpha(Z, X, a, b, X_target):
    N = len(Z) #number of hidden states Z including z0
    K = len(X) #number of hidden states X including x0
    T=len(X_target)

    path = [1] #We suppose at time t=0 we are at Z1
    alpha=np.zeros((N,T+1))
    alpha[path[0]][0]=1


    for t in range(1, T+1):
        for j in range(N):
            sum=0
            for i in range(N):
                sum+= alpha[i][t-1] * a[i][j]

            k=X_target[t-1]
            alpha[j][t] = sum * b[j][k]

        jMax = np.argmax(alpha.T[t])
        path.append(jMax)

    return path, alpha


def probaPath(path, alpha):
    proba = 1
    T=alpha.shape[1]-1

    for t in range(T+1):
        proba *= alpha[path[t]][t]

    return proba

path, alpha = calculAlpha(Z, X, a, b, X_target)

print("path")
print(path)

print()
print("alpha")
print(alpha)

print()
print("proba of the path")
print(probaPath(path, alpha))

