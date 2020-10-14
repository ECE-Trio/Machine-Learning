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

# fit a Gaussian Mixture Model with two components
clf = mixture.GaussianMixture(n_components=3, covariance_type='full')
clf.fit(X_train)

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