import numpy as np
import matplotlib.pyplot as plt

from Dense import Dense
from TanHFunc import Tanh
from LossFuncs import MSE, MSE_prim
from NetworkFuncs import predict, train

X = np.reshape([[0,0],[0,1],[1,0],[1,1]],(4,2,1))
Y = np.reshape([[0],[1],[1],[0]], (4,1,1))

network = [
    Dense(2,3),
    Tanh(),
    Dense(3,1),
    Tanh()
]

train(network, MSE, MSE_prim, X, Y, epochs=10000, alpha=0.1)

points = []
for x in np.linspace(0,1,20):
    for y in np.linspace(0,1,20):
        z = predict(network, [[x], [y]])
        points.append([x,y,z[0,0]])

points = np.array(points)

fig = plt.figure()
ax = fig.add_subplot(111, projection = "3d")
ax.scatter(points[:,0], points[:,1], points[:,2], c=points[:,2], cmap="winter")

plt.show()