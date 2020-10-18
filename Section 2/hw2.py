"""
Made by the team :
    Olivia Dalmasso
    Alexis Direz
    Neil Ségard

Members³ of : ING5 SI DBA GR1.A
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture

n_samples = 300

# generate random sample, two components
np.random.seed(0)

# generate spherical data centered on (20, 20, 20)
shifted_gaussian = np.random.randn(n_samples, 3) + np.array([20, 20, 20])
shifted_gaussian2 = np.random.randn(n_samples, 3) + np.array([12, 13, 14])

# concatenate the two datasets into the final training set
X_train = np.vstack([shifted_gaussian, shifted_gaussian2])

#Normalization of X_train
max=np.max(abs(X_train))
X_train = X_train / max

#Initialization
J=2
I,N=X_train.shape
phi=np.ones(J)*(1/J)
sigma=np.zeros((J,N,N))
mu=np.zeros((J,N))
W=np.zeros((I,J))
tab=np.zeros((I,J))
Wold=np.ones((I,J))

mu[1] += 0.5

for j in range(J):
    for n in range(N):
        while sigma[j][n][n]==0: #to avoid having 0 in the diagonal
            sigma[j][n][n]=np.random.rand()

print("Epoch 0 (init):")
print("\nmu")
print(mu * max)

counter=0
while np.sum(abs(Wold-W)>0.01) != 0:
    counter += 1

    # E-step
    for j in range(J): #for each cluster
        det = 1/ ( (2*np.pi)**(N/2) * np.linalg.det(sigma[j])**0.5)

        for i in range(I):#for each point
            a = (X_train[i] - mu[j]).reshape(3,1)
            b = -0.5 * np.dot(a.T, np.linalg.inv(sigma[j]))
            c = np.dot(b, a)
            d = np.exp(c)

            tab[i][j] = det * d * phi[j]

    Wold=np.copy(W)
    alpha=0
    for i in range(I):
        denom=np.sum(tab[i])

        for j in range(J):
            W[i][j] = (tab[i][j] + alpha) / (denom + alpha*J)

    # M-Step
    Wsomme=np.sum(W,axis=0) #de dim J

    phi=Wsomme/I

    for j in range(J):
        s=0
        for i in range(I):
            s += W[i][j] * X_train[i]
        mu[j] = s / Wsomme[j]

    for j in range(J):
        s=0
        for i in range(I):
            a=(X_train[i] - mu[j]).reshape(3,1)
            s += W[i][j] * np.dot(a, a.T)
        sigma[j]= s / Wsomme[j]

    print()
    print("Epoch:", counter)
    print("mu")
    print(mu * max)


X_train *= max
mu *= max

print()
print()
print("Final mu (in {} epochs):".format(counter))
print(mu)


##Finding radiuses
distances=np.zeros(J)
for i in range(I):
    cluster = (W[i][0] < W[i][1])*1
    dist = np.linalg.norm(X_train[i]-mu[cluster])

    if dist > distances[cluster]:
        distances[cluster]=dist


## Plotting
def WireframeSphere(centre=[0.,0.,0.], radius=1.,
                    n_meridians=20, n_circles_latitude=None):

    if n_circles_latitude is None:
        n_circles_latitude = np.max([n_meridians/2, 4])

    u, v = np.mgrid[0:2*np.pi:n_meridians*1j, 0:np.pi:n_circles_latitude*1j]
    sphere_x = centre[0] + radius * np.cos(u) * np.sin(v)
    sphere_y = centre[1] + radius * np.sin(u) * np.sin(v)
    sphere_z = centre[2] + radius * np.cos(v)
    return sphere_x, sphere_y, sphere_z

#draw sphere
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for j in range(J):
    frame_xs, frame_ys, frame_zs = WireframeSphere(mu[j], distances[j])
    ax.plot_surface(frame_xs, frame_ys, frame_zs, color="g", alpha=0.5)

ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2])
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')


plt.show()

