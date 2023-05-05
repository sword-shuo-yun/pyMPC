#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：python_mpc 
@File    ：MPC_test_wei.py
@Author  ：梁伟
@mail    ：1461334995@qq.com
@Date    ：2023/4/20 16:11 
"""
import matplotlib.pyplot as plt
import scipy.linalg
from cvxopt import solvers, matrix

from pyMPC.kalman import *

# A = np.array([[0.8725, 0, 0, 0],
#               [0, 0.8187, 0, 0],
#               [0, 0, 0.8826, 0],
#               [0, 0, 0, 0.8317]])
# B = np.array([[3.5056, 0],
#               [1.6994, 0],
#               [0, 1.762],
#               [0, 0.8562]])
# C = np.array([[0.1059, 0, -0.0403, 0],
#               [0, 0.0373, 0, -0.0393]])
A = np.array([[1, 1], [-1, 2]])
B = np.array([[1, 1], [1, 2]])
C = np.array([[1, 0], [0, 1]])

D = np.zeros((2, 2))

n = A.shape[0]
p = B.shape[1]
m = C.shape[0]

Q = np.array([[10, 0], [0, 10]])
F = np.array([[1, 0], [0, 1]])
R = np.array([[1, 0], [0, 1]])

r = np.array([[10], [5]])
N = 5


def cal_matrices(A, B, C, D, Q, R, F, N):
    n = A.shape[0]
    p = B.shape[1]
    m = C.shape[0]
    C_bar = np.kron(np.eye(N + 1), C)
    M = np.vstack((np.eye(n), np.zeros((N * n, n))))
    L = np.zeros(((N + 1) * n, N * p))
    temp = np.eye(n)
    for i in range(N):
        rows = i * n + n
        L[rows:rows + n, :] = np.hstack(
                (np.dot(temp, B), L[rows - n: rows, 0:(N - 1) * p]))
        temp = np.dot(temp, A)
        M[rows:rows + n, :] = temp
    Q_bar_be = np.kron(np.eye(N), Q)
    Q_bar = scipy.linalg.block_diag(Q_bar_be, F)
    R_bar = np.kron(np.eye(N), R)
    E = M.T @ C_bar.T @ Q_bar @ C_bar @ L
    H = L.T @ C_bar.T @ Q_bar @ C_bar @ L + R_bar

    return E, H


def prediction(H, E, p):
    sol = solvers.qp(H, E)
    U_thk = np.array(sol['x'])
    u_k = U_thk[0:p, :]
    return u_k


E, H = cal_matrices(A, B, C, D, Q, R, F, N)

C_pinv = np.linalg.pinv(C)
k_step = 100
Y_k = np.zeros((m, k_step))
X_k = np.zeros((n, k_step))
U_k = np.zeros((p, k_step))
Y_k[:, 0] = np.array([10, -10])
u0 = np.array([[[0], [0]]])

# Basic Kalman filter design
Q_kal = 10 * np.eye(n)
R_kal = np.eye(m)
L, P, W = kalman_design_simple(A, B, C, D, Q_kal, R_kal,
                               type = 'predictor')
x0 = np.array([[0], [0]])
x0_est = x0
KF = LinearStateEstimator(x0_est, A, B, C, D, L)
r_k = np.zeros((m, k_step))
r_k[:, 0:1] = r

for k in range(1, k_step):
    if k > 20:
        r = np.array([[5], [0]])
    if k > 40:
        r = np.array([[5], [-10]])
    if k > 60:
        r = np.array([[10], [-5]])
    if k > 80:
        r = np.array([[0], [0]])
    r_k[:, k - 1:k] = r
    y_delta = Y_k[:, k - 1:k] - r
    y_delta = y_delta.reshape(m, 1)
    # y_meas = Y_k[:, k:k + 1] - r
    # x_est_k = X_k[:, k:k + 1]
    x_delta = C_pinv @ y_delta
    T = x_delta.T @ E
    T = matrix(T.T)
    H = matrix(H)
    u_k_delta = prediction(H, T, p)

    # KF.update(y_meas)
    # KF.predict(u_k_delta)

    x_1_delta = A @ x_delta + B @ u_k_delta
    y_1_delta = C @ x_1_delta

    Y_k[:, k:k + 1] = y_1_delta + r.reshape(m, 1)
    u0 = u0 if k < 2 else U_k[:, k - 2:k - 1]
    U_k[:, k - 1:k] = u0 + u_k_delta

fig = plt.figure
ax1 = plt.subplot(211)
ax1.set(ylabel = 'y(t)', xlabel = 't')
for i in range(Y_k.shape[0]):
    ax1.plot([t for t in range(k_step - 1)], Y_k[i, :-1], label = 'y' + str(i))
for i in range(r_k.shape[0]):
    ax1.plot([t for t in range(k_step - 1)], r_k[i, :-1], label = 'r' + str(i))
plt.legend()
ax2 = plt.subplot(212)
ax2.set(ylabel = 'u(t)', xlabel = 't')
for i in range(U_k.shape[0]):
    ax2.plot([t for t in range(k_step - 1)], U_k[i, :-1], label = 'u' + str(i))
plt.legend()
plt.suptitle('MPC Simulink', fontsize = 14)
plt.show()
