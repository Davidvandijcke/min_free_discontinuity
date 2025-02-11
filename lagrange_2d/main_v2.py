import primal_dual
import numpy as np
import pickle 
from skimage import io

# We minimize
# E(u) = \alpha \int \abs{\nabla u}^2 + \int_{J_u} psi(x,u^-,u^+) + \int \rho(x,u)
# where $u:[0,1]^2 \to \R$ is a piecewise smooth function whose jump set is $J_u$.

# The algorithm is the primal-dual algorithm designed by
# Chambolle, Cremers and Strekalovskiy in the article
# "A Convex Representation for the Vectorial Mumford-Shah Functional".
# However, the method also extends to other free-discontinuity functionals,
# provided one adapts the definition of $\alpha$, $\psi$ and $\rho$.

def init_torch():
    import torch
    if torch.cuda.is_available(): # cuda gpus
        device = torch.device("cuda")
        #torch.cuda.set_device(int(gpu_id))
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    elif torch.backends.mps.is_available(): # mac gpus
        device = torch.device("mps")
    elif torch.backends.mkl.is_available(): # intel cpus
        device = torch.device("mkl")
    return device


def init():
    
    dev = init_torch()
    


    # read in image instead
    img = io.imread('camera.png')

    ## Discretization of the domain and the color channel
    N, M = img.shape
    N, M = 32, 32

    ## Precision constants.
    E_algo = 0.0000001
    K_algo = 20000

    ## Operator norm of A
    L_0 = 5 # 5
    #L_0 = np.sqrt(8)
    sigma_0 = 1 / L_0
    tau_0 = 1 / L_0


    ## Dirichlet conditions
    # We define an indicator function S:[0,1]^2 \to {0,1}$ and we define
    # a function U0:[0,1]^2 \to [0,1] such that for all coordinates (i,j),
    # the condition S(i,j) = 1 imposes u(i,j) = U0(i,j).
    S = np.ndarray(shape=(N,N), dtype=np.int8)
    U0 = np.ndarray(shape=(N,N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            if (i == 0) or (i == N-1) or (j == 0) or (j == N-1):
                S[i][j] = 1
                U0[i][j] = 0
            else:
                S[i][j] = 0

    # Definition of the energy (gamma = 0 for the homogeneous functional)
    # Set gamma = 0 for the homogeneous functional
    R0 = 0.25
    alpha, beta, gamma = 1, 5/R0, 1/R0
    # Set the surface energy

    # fill an N x N x M x M array with beta
    psi = np.full((N,N,M,M), beta)

    # psi = np.zeros((N,N,M,M))
    # for i in range(N):
    #     for j in range(N):
    #         for k in range(M):
    #             for l in range(M):
    #                 # Mumford-Shah functional
    #                 psi[i,j,k,l] = beta
                    # Thermal Insulation functional
                    # psi[i,j,k,l] = beta * ((k/M)**2 + (l/M)**2)# Set the error term
    # For the Mumford-Shah functional, we need to define an image f first.
    def f(x,y):
        if np.sqrt((x-1/2)**2 + (y-1/2)**2) < 1/8:
            return 1
        else:
            return 0
    rho = np.empty([N,N,M], dtype=np.float64)

    for i in range(N):
        for j in range(N):
            for k in range(M):
                # Mumford-Shah functional with image as function
                rho[i,j,k] = gamma * ((k+1)/M - f((i+1)/N, (j+1)/N))**2 

                # Mumford-Shah functional with image file
                #rho[i,j,k] = gamma * (k/M - img[i, j])**2 

                # Thermal Insulation functional
                #if k == 1:
                #    rho[i,k] = 0
                #else:
                #    rho[i,k] = gamma**2

    v, sigma = primal_dual.primal_dual(N, M, sigma_0, tau_0, L_0, psi, alpha, rho, U0, S, E_algo, K_algo)
    data = {}
    data["v"] = v
    data["sigma"] = sigma
    data["N"] = N
    data["M"] = M

    with open("data.pkl","wb") as f:
        pickle.dump(data,f)
