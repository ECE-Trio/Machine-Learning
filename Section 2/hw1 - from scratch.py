import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture

n_samples = 300

# generate random sample, two components
np.random.seed(0)

# generate spherical data centered on (20, 20, 20)
shifted_gaussian = np.random.randn(n_samples, 3) + np.array([20, 20, 20])

# generate zero centered stretched Gaussian data
C = np.array([[0., 3, 0], [3.5, .7, 1], [4, 1.2, -0.5]])
stretched_gaussian = np.dot(np.random.randn(n_samples, 3), C)

# concatenate the two datasets into the final training set
X_train = np.vstack([shifted_gaussian, stretched_gaussian])

#Initialization
J=3
N,I=X_train.shape
phi=np.ones(J)*(1/J)
mu=np.zeros((N,J))
sigma=np.zeros((J,N,N))

for j in range(J):
    for n in range(N):
        while sigma[j][n][n]==0: #to avoid having 0 in the diagonal
            sigma[j][n][n]=1#np.random.rand()

## E-step

W=np.zeros((I,J))
tab=np.zeros((I,J))
for j in range(J):
    det = 1/ ( (2*np.pi)**(N/2) * np.linalg.det(sigma[j])**0.5)

    for i in range(I):
        a=(X_train[:,i] - mu[:,j]).reshape(600,1)
        b=-0.5* np.dot(a.T, np.linalg.inv(sigma[j]))
        c= np.dot(b, a)
        d= np.exp(c)

        tab[i][j] = det * d * phi[j]



for i in range(I):
    denom=np.sum(tab[i])

    for j in range(J):
        W[i][j] = tab[i][j] / denom
        W[i][j]=np.random.rand()

## M-Step

Wsomme=np.sum(W,axis=0) #de dim J

phi=Wsomme/I

for j in range(J):
    for i in range(I):
        s += W[i][j] * X_train[:, i]
    mu[:,j] = s / Wsomme[j]

for j in range(J):
    for i in range(I):
        a=(X_train[:,i] - mu[:,j]).reshape(600,1)
        s = W[i][j] * a * a.T
    sigma[j]= s / Wsomme[j]




"""

# new mu optimized
W_sum_j=[0,0,0]

mu_transpose_j = np.shape((J,I))
for j in range(J):
    sum = 0
    sum_omega=0
    for i in range(I):
        sum+=W[i][j]*X_train[i]
        sum_omega+= W[i][j]
    W_sum_j[j]=sum_omega
    mu_transpose_j [j] = sum / W_sum_j[j]
mu = mu_transpose_j.T

#new phi optimized

for j in range(J):
    phi[j]= W_sum_j[j]/I

# new sigma optimized

for j in range(J):
    sum = 0
    for i in range(I):
        sum+=W[i][j] * np.dot(X[i]-mu_transpose_j[j], (X[i]-mu_transpose_j[j]).T )
    sigma[j]= sum / W_sum_j[j]

## Y predict




# display predicted scores by the model as a contour plot
x = np.linspace(-20., 30., num = 7)
y = np.linspace(-20., 40., num = 7)
z = np.linspace(-20., 35., num = 7)
X, Y, Z = np.meshgrid(x, y, z)
XXX = np.array([X.ravel(), Y.ravel(), Z.ravel()]).T
S = -clf.score_samples(XXX)
S = S.reshape(X.shape)

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter(X, Y, Z)

#CB = plt.colorbar(CS, shrink=0.8, extend='both')

ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2])

plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
"""