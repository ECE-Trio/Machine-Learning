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
N = 4 #number of hidden states Z including z0
K = 5 #number of hidden states X including x0

a_ij = np.zeros((N,N))
b_jk = np.zeros((N,K))

X= [1,3,2,0]
T=len(X)

path = [1] #We suppose at time t=0 we are at Z1

def alpha_j(j,t):
    sum=0
    if j == 1 and t==0:
        return 1
    elif (j != 1 and t==0):
        return 0
    else:
        for i in range(N):
            sum+= alpha_j(i,t-1)*a_ij[i,j]*b_jk[j,k]  # comment choisir le k ?
