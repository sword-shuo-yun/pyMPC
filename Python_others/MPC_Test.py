import numpy as np

from cvxopt import matrix
import matplotlib.pyplot as plt
 
from MPC_Matrices import MPC_Matrices
from Prediction import Prediction


A = np.array([[1, 1], [-1, 2]])
n = A.shape[0]

B = np.array([[1, 1],[1, 2]])
p = B.shape[1]

Q = np.array([[1, 0],[0, 1]])
F = np.array([[1, 0],[0, 1]])
R = np.array([[1, 0],[0, 0.1]])

k_steps = 100

X_k = np.zeros((n, k_steps))

X_k[:,0] = [20, -20]

print("X_k_init",X_k)

U_k = np.zeros((p, k_steps))


N = 5


M, C = MPC_Matrices(A,B,Q,R,F,N)
M = matrix(M)



for k in range(1,k_steps):
    x_kshort = X_k[:, k - 1].reshape(2, 1)
    u_kshort = U_k[:, k - 1].reshape(2, 1)
    T = np.dot(C,x_kshort)
    T = matrix(T)
    for i in range(2):
        U_k[i:,k-1] = Prediction(M,T)[i,0]

    X_knew = np.dot(A,x_kshort) + np.dot(B,u_kshort)

    for j in range(2):
        X_k[j:,k] = X_knew[j,0]





print(X_k)
X_k_index_array = []
U_k_index_array = []
# X1_array = []
# X2_array = []
for i in range(len(X_k[0])):
    X_k_index_array.append(i)
 
for m in range(len(U_k[0])):
    U_k_index_array.append(m)

# plot
plt.subplot(2,1,1)       
plt.plot(X_k_index_array, X_k[0], label='x1') 
plt.plot(X_k_index_array, X_k[1], label='x2') 
plt.legend() 

plt.subplot(2,1,2)       
plt.plot(U_k_index_array, U_k[0], label='u1') 
plt.plot(U_k_index_array, U_k[1], label='u2') 
plt.legend() 

# plt.xlabel('Plot Number')                 
# plt.ylabel('Important var')                
plt.grid(linestyle = "--")                                
plt.show()   
