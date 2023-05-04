import numpy as np
import matplotlib.pyplot as plt
import MPC_Matrices
import Prediction


def test_02():
    A = np.array([[1,0.1],[0,2]])
    n = A.shape[1-1]
    B = np.array([[0], [0.5]])
    p = B.shape[2 - 1]
    k_steps = 100
    X_K = np.zeros((n,k_steps))

    # 将X_K[:,1]修改为X_K[:,[1]]
    X_K[:,[0]] = np.array([[20],[- 20]])
    U_K = np.zeros((p,k_steps))
    print()


def test_day_01():
    A = np.array([[1,0.1],[0,2]])
    n = A.shape[1-1]
    B = np.array([[0],[0.5]])
    p = B.shape[2-1]
    Q = np.array([[1,0],[0,1]])
    F = np.array([[1,0],[0,1]])
    R = np.array([0.1])
    k_steps = 100
    X_K = np.zeros((n,k_steps))
    X_K[:,[0]] = np.array([[20],[- 20]])
    U_K = np.zeros((p,k_steps))
    N = 5
    E,H = MPC_Matrices.MPC_Matrices(A,B,Q,R,F,N)


    for k in range(1,k_steps):
        U_K[:,[k-1]] = Prediction.Prediction(X_K[:,[k-1]],E,H,N,p)

        A_tmp = np.dot(A, X_K[:,k-1])
        B_tmp = np.dot(B, U_K[:,k-1])
        X_K[:, k] = A_tmp + B_tmp

    print(X_K)
    X_k_index_array = []
    U_k_index_array = []
    # X1_array = []
    # X2_array = []
    for i in range(len(X_K[0])):
        X_k_index_array.append(i)

    for m in range(len(U_K[0])):
        U_k_index_array.append(m)

    # plot
    plt.subplot(2, 1, 1)
    plt.plot(X_k_index_array, X_K[0], label='x1')
    plt.plot(X_k_index_array, X_K[1], label='x2')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(U_k_index_array, U_K[0], label='u1')
    # plt.plot(U_k_index_array, U_K[1], label='u2')
    plt.legend()

    # plt.xlabel('Plot Number')
    # plt.ylabel('Important var')
    plt.grid(linestyle="--")
    plt.show()


if __name__ == '__main__':
    test_day_01()
    # test_02()
