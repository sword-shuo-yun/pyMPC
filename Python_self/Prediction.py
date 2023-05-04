import numpy as np
import quadprog
    
def Prediction(x_k = None,E = None,H = None,N = None,p = None): 

    U_k = np.zeros((N * p, 1))
    U_k = quadprog.quadprog(H,np.dot(E,x_k))

    u_k = U_k[p-1,:]
    # u_k = U_k_tmp[:,1]

    return u_k