import numpy as np


def adjConcat(a, b):
    '''
    将a,b两个矩阵沿对角线方向斜着合并，空余处补零[a,0.0,b]
    得到a和b的维度，先将a和b*a的零矩阵按行（竖着）合并得到c，再将a*b的零矩阵和b按行合并得到d
    将c和d横向合并
    '''
    lena = len(a)
    lenb = len(b)
    left = np.row_stack((a, np.zeros((lenb, lena))))  # 先将a和一个len(b)*len(a)的零矩阵垂直拼接，得到左半边
    right = np.row_stack((np.zeros((lena, lenb)), b))  # 再将一个len(a)*len(b)的零矩阵和b垂直拼接，得到右半边
    result = np.hstack((left, right))  # 将左右矩阵水平拼接
    return result


def MPC_Matrices(A = None,B = None,Q = None,R = None,F = None,N = None): 
    n = A.shape[1-1]
    p = B.shape[2-1]
    M = np.vstack((np.eye(n),np.zeros((N * n,n))))
    C = np.zeros(((N + 1) * n,N * p))
    tmp = np.eye(n)
    for i in np.arange(1,N+1).reshape(-1):
        rows = i * n + (np.arange(0, n))
        C_tmp1 = C[rows - n, :]
        C_tmp = C_tmp1[:, np.arange(0, C.shape[1] - p)]
        B_tmp = np.dot(tmp, B)
        C[rows, :] = np.hstack((B_tmp,C_tmp))
        print(C)
        tmp = np.dot(A, tmp)
        M[rows,:] = tmp
    
    Q_bar = np.kron(np.eye(N),Q)
    Q_bar = adjConcat(Q_bar,F)
    R_bar = np.kron(np.eye(N),R)
    E1 = np.dot(np.transpose(C), Q_bar)
    E = np.dot(E1, M)
    H1 = np.dot(np.transpose(C), Q_bar)
    H2 = np.dot(H1, C)
    H = H2+ R_bar
    return E,H