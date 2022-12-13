# Orthogonal projection onto C
# Lagrange multipliers !!

import numpy as np 

## Two-dimensional case

# v = np.zeros((N,N,M)), xi = np.zeros((N,N,M,M+1,2)), eta = np.zeros((N,N,M,M+1))

def projection_C(v,xi,eta,U0,S,N,M):
    # for i in range(0,N):
    #     for j in range(0,N):
    #         v[i,j,0] = 1
    #         v[i,j,M-1] = 0
    #         if S[i,j] == 1: # Dirichlet condition (u = U0 on S)
    #             for k in range(2,M):
    #                 if k/M < U0[i,j]:
    #                     v[i,j,k-1] = 1
    #                 else:
    #                     v[i,j,k-1] = 0
    #         else:
    #             for k in range(1,M-1):
    #                 v[i,j,k] = min(1,max(0,v[i,j,k]))
                  
    # Set v[i,j,0] and v[i,j,M-1] for all i, j
    v[:,:,0] = 1
    v[:,:,M-1] = 0

    # Define array of values for k
    k = np.arange(2, M)
    
    # Set v[i,j,k] for Dirichlet condition (u = U0 on S)
    dirichlet_mask = S == 1
    v[dirichlet_mask, 1:M-1]  = (k/M < U0[dirichlet_mask, None])

    # Set v[i,j,k] for all other k
    v[~dirichlet_mask, 1:M-1] = np.minimum(1, np.maximum(0, v[~dirichlet_mask, 1:M-1]))


    # # Orthogonal projection onto the set { norm(xi) <= eta }
    # for i in range(0,N):
    #     for j in range(0,N):
    #         for k in range(0,M):
    #             for l in range(0,M+1):
    #                 xi_norm = np.sqrt(xi[i,j,k,l,0]**2 + xi[i,j,k,l,1]**2)
    #                 if eta[i,j,k,l] < xi_norm:
    #                     if eta[i,j,k,l] <= - xi_norm:
    #                         eta[i,j,k,l] = 0
    #                         xi[i,j,k,l,0] = 0
    #                         xi[i,j,k,l,1] = 0
    #                     else:
    #                         eta[i,j,k,l] = 1/2 * (eta[i,j,k,l] + xi_norm)
    #                         xi[i,j,k,l,0] = eta[i,j,k,l] * xi[i,j,k,l,0] / xi_norm
    #                         xi[i,j,k,l,1] = eta[i,j,k,l] * xi[i,j,k,l,1] / xi_norm
    
     
    # compute xi_norm for all values of i, j, k, and l
    xi_norm = np.sqrt(xi[:,:,:,:,0]**2 + xi[:,:,:,:,1]**2)

    
    # mask for eta < xi_norm and eta > -xi_norm
    mask1 = (eta < xi_norm) & (eta > -xi_norm)
    mask2 = (eta < xi_norm) & (eta <= -xi_norm)

    # set eta to 0 for values where eta is less than -xi_norm
    eta[mask2] = 0

    # set eta to 1/2 * (eta + xi_norm) for values where mask is True
    eta[mask1] = 1/2 * (eta[mask1] + xi_norm[mask1])

    # set xi to 0 for values where eta is less than -xi_norm
    xi[:,:,:,:,0][mask2] = 0
    xi[:,:,:,:,1][mask2] = 0

    # set xi to eta * xi / xi_norm for values where mask is True
    xi[:,:,:,:,0][mask1] = eta[mask1] * xi[:,:,:,:,0][mask1] / xi_norm[mask1]
    xi[:,:,:,:,1][mask1] = eta[mask1] * xi[:,:,:,:,1][mask1] / xi_norm[mask1]


    # I think by setting it to 0 in the first condition, the second one doesn't override it

    return v, xi, eta

# Orthogonal projection onto
#     {(y1,y2,y3) | y3 >= a1 y1^2 + a2 y2^2 + c },
# where (y1,y2,y3) are orthonormal coordinates.
# The idea is that if y is the orthogonal projection of x,
# then there exists a scalar t such that
#     (y1 - x1, y2 - x2, y3 - x3) = t (2 a1 y1, 2a2 y2, -1)
# and since y3 = a1 y1^2 + a2 y2^2 + c, we deduce an equation f(t) = 0.



# def projection_parabola(x1, x2, x3, a1, a2, c, E_parabola = 0.000001, K_parabola = 15):
#     if x3 < a1 * x1**2 + a2 * x2**2 + c - E_parabola:
#         k = 0
#         t = 0
#         ft = a1 * (x1)**2 + a2 * x2**2 + c - x3 
#         while abs(ft) > E_parabola and k < K_parabola:
#             k = k + 1
#             ft = t + a1 * (x1 / (1 - 2 * a1 * t))**2 + a2 * (x2 / (1 - 2 * a2 * t))**2 + c - x3 
#             dft = 1 + (4 * a1**2 * x1**2) / (1 - 2 * a1 * t)**3 + (4 * a2**2 * x2**2) / (1 - 2 * a2 * t)**3
#             t = t - ft/dft
#         x1 = x1 / (1 - 2 * a1 * t)
#         x2 = x2 / (1 - 2 * a2 * t)
#         x3 = x3 - t
#     return x1, x2, x3

def projection_parabola_vectorized(x1, x2, x3, a1, a2, c, E_parabola = 0.000001, K_parabola = 15):
    x1, x2, x3 = np.array(x1), np.array(x2), np.array(x3)
    t = np.zeros(len(x1))
    ft = a1 * (x1)**2 + a2 * x2**2 + c - x3
    k = 0
    while np.any(np.abs(ft) > E_parabola) and (k < K_parabola):
        k = k + 1
        ft = t + a1 * (x1 / (1 - 2 * a1 * t))**2 + a2 * (x2 / (1 - 2 * a2 * t))**2 + c - x3 
        dft = 1 + (4 * a1**2 * x1**2) / (1 - 2 * a1 * t)**3 + (4 * a2**2 * x2**2) / (1 - 2 * a2 * t)**3
        t = t - ft/dft
    x1 = x1 / (1 - 2 * a1 * t)
    x2 = x2 / (1 - 2 * a2 * t)
    x3 = x3 - t
    return x1, x2, x3




def projection_K(sigma,alpha, rho, N, M):
    # a = 1 / (4 * alpha)

    # for i in range(0,N):
    #     for j in range(0,N):
    #         for k in range(0,M):
    #             sigma[i,j,k,0], sigma[i,j,k,1], sigma[i,j,k,2] = projection_parabola(sigma[i,j,k,0],sigma[i,j,k,1],sigma[i,j,k,2],a,a,-rho[i,j,k])
                
    # return sigma
    
    a = 1 / (4 * alpha)
    sigma = sigma.reshape(N, N, M, 3)
    x1, x2, x3 = sigma[:, :, :, 0], sigma[:, :, :, 1], sigma[:, :, :, 2]
    x1, x2, x3 = projection_parabola_vectorized(x1, x2, x3, a, a, -rho)
    sigma[:, :, :, 0], sigma[:, :, :, 1], sigma[:, :, :, 2] = x1, x2, x3
    return sigma.reshape(N, N, M, 3)



import numpy as np
