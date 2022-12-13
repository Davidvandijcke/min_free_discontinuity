
import numpy as np 
from operators import *
from projections import *



def primal_dual(N, M, sigma_0, tau_0, L_0, psi, alpha, rho, U0, S, E_algo = 1e-6, K_algo = 1000):
    v = np.zeros((N,N,M))
    xi = np.zeros((N,N,M,M+1,2))
    eta = np.zeros((N,N,M,M+1))
    mu = np.zeros((N,N,M,2))
    projection_C(v,xi,eta,U0,S,N,M)

    v_next = np.zeros((N,N,M))
    xi_next = np.zeros((N,N,M,M+1,2))
    eta_next = np.zeros((N,N,M,M+1))
    mu_next = np.zeros((N,N,M,2))

    v_bar = np.copy(v)
    xi_bar = np.copy(xi)
    eta_bar = np.copy(eta)
    mu_bar = np.copy(mu)

    sigma = np.zeros((N,N,M,3)) 
    m = np.zeros((N,N,M,M+1))
    p = np.zeros((N,N,M+1,2))
    # for i in range(N):
    #     for j in range(N):
    #         for k in range(M):
    #             for l in range(k+2,M+1): # stays same as in julia cause it's relative
    #                 m[i,j,k,l] = psi[i,j,k,l-1] * M
    
    for k in range(M):
        l_indices = np.arange(k+2,M+1)
        m[:,:,k,l_indices] = psi[:,:,k,l_indices-1] * M

    count = 0
    error = 1
    energy = E_operator(v_next,xi_next,eta_next,mu_next,sigma,m,p,N,M)

    while error > E_algo and count < K_algo:
        A_sigma, A_m, A_p = A_operator(v_bar,xi_bar,eta_bar,mu_bar,N,M)
        sigma = projection_K(sigma + sigma_0 * A_sigma, alpha, rho, N, M)
        m = m + sigma_0 * A_m
        # for i in range(N):
        #     for j in range(N):
        #         for k in range(M):
        #             # for l in range(k+2,M+1):
        #             #     if m[i,j,k,l] > psi[i,j,k,l-1] * M:
        #             #         m[i,j,k,l] = psi[i,j,k,l-1] * M
                            
        #             # Create an array of indices for l
        #             l_indices = np.arange(k+2, M+1)

        #             # Calculate the element-wise minimum of m and psi * M,
        #             # but only for the indices specified by l_indices
        #             m[i,j,k,l_indices] = np.minimum(m[i,j,k,l_indices], psi[i,j,k,l_indices-1] * M)


        # Calculate the element-wise minimum of m and psi * M,
        # but only for the indices specified by l_indices
        for k in range(M):
            l_indices = np.arange(k+2, M+1)
            m[:,:,k,l_indices] = np.minimum(m[:,:,k,l_indices], psi[:,:,k,l_indices-1] * M)
                            


        p = p + sigma_0 * A_p

        B_v, B_xi, B_eta, B_mu = B_operator(sigma,m,p,N,M)
        v_next, xi_next, eta_next = projection_C(v - tau_0 * B_v, xi - tau_0 * B_xi, eta - tau_0 * B_eta, U0,S,N,M)
        mu_next =  mu - tau_0 * B_mu

        v_bar = 2 * v_next - v
        xi_bar = 2 * xi_next - xi
        eta_bar = 2 * eta_next - eta
        mu_bar = 2 * mu_next - mu

        energy_next = E_operator(v_next,xi_next,eta_next,mu_next,sigma,m,p,N,M)
        error = abs(energy_next - energy)
        count = count + 1

        energy = energy_next
        v = np.copy(v_next)
        xi = np.copy(xi_next)
        eta = np.copy(eta_next)
        mu = np.copy(mu_next)
        
        # k modulo 50
  
        print("count = ", count, "energy = ", energy, "error = ", error)
    return v, sigma