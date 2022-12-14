
using Plots
using JLD
gr()

import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import math

data = scipy.io.loadmat("./data.pkl")

v = data["v"]
sigma = data["sigma"]
N = data["N"]
M = data["M"]

I = np.zeros((N,1))
J = np.zeros((N,1))
for i in range(1,N):
    I[i] = i/N

for k in range(1,M):
    J[k] = k/M

#1D
u = np.zeros((N,1))
for i in range(1,N):
    k = 1
    while k < M and v[i,k] > 0.5:
        k = k + 1
    u[i] = k/M

plt.plot(I,u)
plt.xlim((1/M,1))
plt.ylim((1/M,1))
plt.show()

#2D
u = zeros(N,N);
for i = 1:N
   for j = 1:N
       k = 1
       while k < M && v[i,j,k] > 0.5
           k = k + 1
       end
       u[i,j] = k/M
   end
end
p = plot(I,I,u,seriestype=:surface,xlim=(1/M,1),ylim=(1/M,1),zlim=(1/M,1))
display(p)