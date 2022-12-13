# The bilinear operator E and two linear operators A, B such that
# E(v,xi,mu,sigma,m,p) = <A(v,xi,mu),(sigma,m,p)> = <(v,xi,mu),B(sigma,m,p)>.

import numpy as np 



# v = np.ndarray((N,N,M)), xi = np.ndarray((N,N,M,M+1,2)), eta = np.ndarray((N,N,M,M+1)), mu = np.ndarray((N,N,M,2)),
# sigma = np.ndarray((N,N,M,3)), m = np.ndarray((N,N,M,M+1)), p = np.ndarray((N,N,M+1,2)).

# def E_operator(v,xi,eta,mu,sigma,m,p, N, M):
#     E = 0
#     for i in range(N):
#         for j in range(N):
#             for k in range(M):
#                 if i < (N - 1):
#                     E = E + (v[i+1,j,k] - v[i,j,k]) * sigma[i,j,k,0]
#                 if j < (N - 1):
#                     E = E + (v[i,j+1,k] - v[i,j,k]) * sigma[i,j,k,1]
#                 if k < (M - 1):
#                     E = E + (v[i,j,k+1] - v[i,j,k]) * sigma[i,j,k,2]
#                 E = E + mu[i,j,k,0] * (p[i,j,k+1,0] - p[i,j,k,0] - sigma[i,j,k,0])
#                 E = E + mu[i,j,k,1] * (p[i,j,k+1,1] - p[i,j,k,1] - sigma[i,j,k,1])
#             for k in range(M):
#                 for l in range(k+2,M+1):
#                     E = E + xi[i,j,k,l,0] * (p[i,j,l,0] - p[i,j,k,0])
#                     E = E + xi[i,j,k,l,1] * (p[i,j,l,1] - p[i,j,k,1])
#                     E = E + eta[i,j,k,l] * m[i,j,k,l]
#     return E


def E_operator(v, xi, eta, mu, sigma, m, p, N, M):
    # Create an empty NumPy array with the same shape as v
    # This array will store the result of the E operator
    E = np.zeros_like(v)

    # Compute the first part of the E operator
    E[:-1,:,:] += sigma[:N-1,:,:,0] * (v[1:,:,:] - v[:-1,:,:])
    E[:,:-1,:] += sigma[:,:N-1,:,1] * (v[:,1:,:] - v[:,:-1,:])
    E[:,:,:-1] += sigma[:,:,:M-1,2] * (v[:,:,1:] - v[:,:,:-1])

    # Compute the second part of the E operator
    E[:,:,:] += mu[:,:,:,0] * (p[:,:,1:,0] - p[:,:,:-1,0] - sigma[:,:,:,0])
    E[:,:,:] += mu[:,:,:,1] * (p[:,:,1:,1] - p[:,:,:-1,1] - sigma[:,:,:,1])

    # Compute the third part of the E operator
    for k in range(M):
        for l in range(k+2, M+1):
            E[:,:,k] += xi[:,:,k,l,0] * (p[:,:,l,0] - p[:,:,k,0])
            E[:,:,k] += xi[:,:,k,l,1] * (p[:,:,l,1] - p[:,:,k,1])
            E[:,:,k] += eta[:,:,k,l] * m[:,:,k,l]
    return np.sum(E)

                    

# def A_operator(v, xi, eta, mu, N, M):
#     sigma = np.zeros((N,N,M,3))
#     m = np.zeros((N,N,M,M+1))
#     p = np.zeros((N,N,M+1,2))
#     for i in range(N):
#         for j in range(N):
#             for k in range(M):
#                 if i < (N-1):
#                     sigma[i,j,k,0] = v[i+1,j,k] - v[i,j,k]
#                 elif i == (N-1):
#                     sigma[i,j,k,0] = - v[i,j,k]
#                 if j < (N-1):
#                     sigma[i,j,k,1] = v[i,j+1,k] - v[i,j,k]
#                 elif j == (N-1):
#                     sigma[i,j,k,1] = - v[i,j,k]
#                 if k < (M-1):
#                     sigma[i,j,k,2] = v[i,j,k+1] - v[i,j,k]
#                 elif k == M-1:
#                     sigma[i,j,k,2] = - v[i,j,k]
#                 sigma[i,j,k,0] = sigma[i,j,k,0] - mu[i,j,k,0]
#                 sigma[i,j,k,1] = sigma[i,j,k,1] - mu[i,j,k,1]
#             for k in range(M+1):
#                 for l in range(M+1):
#                     if l <= (k-2):
#                         p[i,j,k,0] = p[i,j,k,0] + xi[i,j,l,k,0]
#                         p[i,j,k,1] = p[i,j,k,1] + xi[i,j,l,k,1]
#                     elif l >= (k+2):
#                         p[i,j,k,0] = p[i,j,k,0] - xi[i,j,k,l,0]
#                         p[i,j,k,1] = p[i,j,k,1] - xi[i,j,k,l,1]
#                         m[i,j,k,l] = eta[i,j,k,l]
#                 if k == 0:
#                     p[i,j,k,0] = p[i,j,k,0] - mu[i,j,k,0]
#                     p[i,j,k,1] = p[i,j,k,1] - mu[i,j,k,1]
#                 elif k == M:
#                     p[i,j,k,0] = p[i,j,k,0] + mu[i,j,k-1,0]
#                     p[i,j,k,1] = p[i,j,k,1] + mu[i,j,k-1,1]
#                 else:
#                     p[i,j,k,0] = p[i,j,k,0] + mu[i,j,k-1,0] - mu[i,j,k,0]
#                     p[i,j,k,1] = p[i,j,k,1] + mu[i,j,k-1,1] - mu[i,j,k,1]
#     return sigma, m, p


def A_operator(v, xi, eta, mu, N, M):
    sigma = np.zeros((N,N,M,3))
    m = np.zeros((N,N,M,M+1))
    p = np.zeros((N,N,M+1,2))
    
    sigma[:-1,:,:,0] = v[1:,:,:] - v[:-1,:,:]
    sigma[-1,:,:,0] = -v[-1,:,:]
    sigma[:,:-1,:,1] = v[:,1:,:] - v[:,:-1,:]
    sigma[:,-1,:,1] = -v[:,-1,:]
    sigma[:,:,:-1,2] = v[:,:,1:] - v[:,:,:-1]
    sigma[:,:,-1,2] = -v[:,:,-1]
    
    sigma[:,:,:,0] -= mu[:,:,:,0]
    sigma[:,:,:,1] -= mu[:,:,:,1]
    
    for k in range(M):
        for l in range(M+1):
            if l <= (k-2):
                p[:,:,k,0] += xi[:,:,l,k,0]
                p[:,:,k,1] += xi[:,:,l,k,1]
            elif l >= (k+2):
                p[:,:,k,0] -= xi[:,:,k,l,0]
                p[:,:,k,1] -= xi[:,:,k,l,1]
                m[:,:,k,l] = eta[:,:,k,l]
    p[:,:,0,0] -= mu[:,:,0,0]
    p[:,:,0,1] -= mu[:,:,0,1]
    p[:,:,M,0] += mu[:,:,M-1,0]
    p[:,:,M,1] += mu[:,:,M-1,1]
    for k in range(1,M):
        p[:,:,k,0] += mu[:,:,k-1,0] - mu[:,:,k,0]
        p[:,:,k,1] += mu[:,:,k-1,1] - mu[:,:,k,1]
    
    return sigma, m, p




# # rewrite the B_operator above with correct Python indexing, not Julia
# def B_operator(sigma, m, p, N, M):
#     v = np.zeros((N, N, M))
#     xi = np.zeros((N, N, M, M+1, 2))
#     eta = np.zeros((N, N, M, M+1))
#     mu = np.zeros((N, N, M, 2))
#     for i in range(N):
#         for j in range(N):
#             for k in range(M):
#                 if i == 0:
#                     v[i, j, k] = -sigma[i, j, k, 0]
#                 elif i == (N-1):
#                     v[i, j, k] = sigma[i - 1, j, k, 0]
#                 else:
#                     v[i, j, k] = sigma[i-1, j, k, 0] - sigma[i, j, k, 0]
#                 if j == 0:
#                     v[i, j, k] = v[i, j, k] - sigma[i, j, k, 1]
#                 elif j == (N-1):
#                     v[i, j, k] = v[i, j, k] + sigma[i, j - 1, k, 1]
#                 else:
#                     v[i, j, k] = v[i, j, k] + sigma[i, j - 1, k, 1] - sigma[i, j, k, 1]
#                 if k == 0:
#                     v[i, j, k] = v[i, j, k] - sigma[i, j, k, 2]
#                 elif k == (M-1):
#                     v[i, j, k] = v[i, j, k] + sigma[i, j, k - 1, 2]
#                 else:
#                     v[i, j, k] = v[i, j, k] + sigma[i, j, k - 1, 2] - sigma[i, j, k, 2]
#                 for l in range(k + 2, M + 1):
#                     xi[i, j, k, l, 0] = xi[i, j, k, l, 0] + p[i, j, l, 0] - p[i, j, k, 0]
#                     xi[i, j, k, l, 1] = xi[i, j, k, l, 1] - p[i, j, l, 1] - p[i, j, k, 1]
#                     eta[i, j, k, l] = m[i, j, k, l]
#                 mu[i, j, k, 0] = p[i, j, k + 1, 0] - p[i, j, k, 0] - sigma[i, j, k, 0]
#                 mu[i, j, k, 1] = p[i, j, k + 1, 1] - p[i, j, k, 1] - sigma[i, j, k, 1]
#     return v, xi, eta, mu




import numpy as np
def B_operator(sigma, m, p, N, M):
    v = np.zeros((N, N, M))
    xi = np.zeros((N, N, M, M+1, 2))
    eta = np.zeros((N, N, M, M+1))
    mu = np.zeros((N, N, M, 2))
    # calculate v
    v[1:,:,:] = sigma[:-1, :, :, 0] - sigma[1:, :, :, 0] 
    v[:,1:,:] += sigma[:, :-1, :, 1] - sigma[:, 1:, :, 1] 
    v[:,:,1:] += sigma[:, :, :-1, 2] - sigma[:, :, 1:, 2]
    v[0,:,:] -= sigma[0, :, :, 0]
    v[:,0,:] -= sigma[:, 0, :, 1]
    v[:,:,0] -= sigma[:, :, 0, 2]
    v[-1,:,:] += sigma[-1, :, :, 0]
    v[:,-1,:] += sigma[:, -1, :, 1]
    v[:,:,-1] += sigma[:, :, -1, 2]
    # calculate xi
    xi[:,:,:,2:,0] = p[:,:,2:,0] - p[:,:,:-2,0]
    xi[:,:,:,2:,1] = -p[:,:,2:,1] - p[:,:,:-2,1]
    # calculate eta
    eta[:,:,:,2:] = m[:,:,:,2:]
    # calculate mu
    mu[:,:,:,0] = p[:,:,1:,0] - p[:,:,:-1,0] - sigma[:,:,:,0]
    mu[:,:,:,1] = p[:,:,1:,1] - p[:,:,:-1,1] - sigma[:,:,:,1]
    return v, xi, eta, mu