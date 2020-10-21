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

    return path

print(calculAlpha(Z, X, a, b, X_target))





def alpha_j(j,t):
    sum=0
    if j == 1 and t==0:
        return 1
    elif (j != 1 and t==0):
        return 0
    else:
        for i in range(N):
            sum+= alpha_j(i,t-1)*a_ij[i,j]*b_jk[j,k]  # comment choisir le k ?
