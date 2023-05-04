import numpy as np
import scipy.linalg

def MPC_Matrices(A, B, Q, R, F, N):

    n = A.shape[0]
    p = B.shape[1]

    M = np.vstack((np.eye((n)), np.zeros((N*n,n))))
    C = np.zeros(((N+1)*n,N*p))
    tmp = np.eye(n)

    for i in range(N):
        rows = i * n + n
        C[rows:rows+n,:] = np.hstack((np.dot(tmp, B), C[rows-n:rows, 0:(N-1)*p]))
        tmp = np.dot(A, tmp)
        M[rows:rows+n,:] = tmp

    Q_bar_be = np.kron(np.eye(N), Q)
    Q_bar = scipy.linalg.block_diag(Q_bar_be, F)
    R_bar = np.kron(np.eye(N), R)

    G = np.matmul(np.matmul(M.transpose(),Q_bar),M)
    E = np.matmul(np.matmul(C.transpose(),Q_bar),M)
    H = np.matmul(np.matmul(C.transpose(),Q_bar),C) + R_bar

    return H, E