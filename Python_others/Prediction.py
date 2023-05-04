import numpy as np
from cvxopt import solvers

def Prediction(M,T):

    sol = solvers.qp(M,T)
    U_thk = np.array(sol["x"])
    u_k = U_thk[0:2, :]

    return u_k