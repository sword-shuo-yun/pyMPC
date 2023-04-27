import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.sparse as sparse
import time
import control
from cvxpy import Variable, Parameter, Minimize, Problem, OSQP, quad_form
from pyMPC.kalman import kalman_design_simple, LinearStateEstimator



if __name__ == "__main__":

    len_sim = 100  # simulation length (s)

    # Discrete time model of a frictionless mass (pure integrator)
    Ts = 1.0
    # r_den = 0.9  # magnitude of poles
    # wo_den = 0.2  # phase of poles (approx 2.26 kHz)
    #
    # # Build a second-order discrete-time dynamics with dcgain=1 (inner loop model)
    # H_sys = control.TransferFunction([1], [1, -2 * r_den * np.cos(wo_den), r_den ** 2], Ts)
    # H_sys = H_sys / control.dcgain(H_sys)
    # H_ss = control.ss(H_sys)
    #
    # # SISO ABCD
    # Ad = np.array(H_ss.A)
    # Bd = np.array(H_ss.B)
    # Cd = np.array(H_ss.C)
    # Dd = np.array(H_ss.D)
    #
    #
    #
    # # MIMO ABCD
    # Ad = scipy.linalg.block_diag(Ad, Ad)
    # Bd = scipy.linalg.block_diag(Bd, Bd)
    # Cd = scipy.linalg.block_diag(Cd, 1.5*Cd)
    # Dd = scipy.linalg.block_diag(Dd, Dd)


    # Ad = np.array([[-0.004545,0,0,0,0,0], [0,-0.009615,0,0,0,0],[0,0,-0.004161,0,0,0], [0,0,0,-0.006141,0,0], [0,0,0,0,-0.1,0], [0,0,0,0,0,-0.1]])
    # Bd = np.array([[0.125,0 ,0  ,0],[0.0625    , 0   ,  0  ,   0],[0  , 0.0625 ,     0  ,   0],[0 , 0.03125  ,  0,  0],[0,   0, 0.125  ,   0],[0    , 0   ,  0  ,  0.125]])
    # Cd = np.array([[0.1091, 0 ,-0.04527 ,0 ,-0.08, -0.08],[0, 0.05385, 0, -0.01769 , -0.08 ,-0.08]])
    # Dd = np.array([[0 ,0 ,0, 0 ],[ 0, 0 ,0 ,0]])

    Ad = np.array([[-0.02,0,0,0,0,0], [0,-0.01111,0,0,0,0],[0,0,-0.0125,0,0,0], [0,0,0,-0.01667,0,0], [0,0,0,0,-1,0], [0,0,0,0,0,-0.005]])
    Bd = np.array([[0.03125, 0 ],[0 ,  0.03125 ],[ 0.125 ,  0  ],[0 , 0.0625 ],[0.5,   0.5],[ 0  ,  0.125]])
    Cd = np.array([[0.0256, -0.04444,0,0,0,0],[0, 0, 0.1, -0.048, 0, 0], [0, 0, 0, 0, 2, 0] , [0 ,0, 0, 0, 0, 0.2496]])
    Dd = np.array([[0 ,0] ,[0, 0 ],[ 0, 0] ,[0 ,0]])


    [nx, ng] = Bd.shape  # number of states and number or inputs
    [ny, _] = Cd.shape  # number of outputs

    # Constraints
    # 控制输入初始值ginit
    # ginit = np.array([95,35])  #
    ginit = np.array([2, 2])
    gmin = np.array(ng*[-10000.0]) #- gref
    gmax = np.array(ng*[10000.0]) #- gref

    ymin = np.array(ny*[-10000.0])
    ymax = np.array(ny*[10000.0])

    Dgmin = np.array(ng*[-2e-1])
    Dgmax = np.array(ng*[2e-1])


    # Objective function
    Qy = np.diag(ny*[20])   # or sparse.diags([])
    #QyN = np.diag(2*[20])  # final cost
    QDy = np.eye(ny)
    Qrg = 100*np.eye(ng)
    QDg = 0.5 * sparse.eye(ng)  # Quadratic cost for Du0, Du1, ...., Du_N-1

    # Initial and reference
    # 初始状态可测？初始状态xint都为0
    # 如何让y的初始值考虑进来
    x0 = np.array(nx*[0.0])  # initial state
    # y0 = np.array([63, 154, 125, -580])
    y0 = np.array([50, 50, 50, 50])

    # Cd_1 = np.linalg.pinv(Cd)
    # x0 = np.dot(Cd_1,y0)


    # Prediction horizon
    Np = 40

    # Define problem
    g = Variable((ng, Np))
    x = Variable((nx, Np))
    x_init = Parameter(nx)
    gminus1 = Parameter(ng)
    yminus1 = Parameter(ny)
    r_y = Parameter(ny)
    r_u = Parameter(ng)

    objective = 0.0
    constraints = [x[:, 0] == x_init]

    y = Cd @ x + Dd @g


    for k in range(Np):
        objective += quad_form(y[:, k] - r_y, Qy)   # tracking cost
        objective += quad_form(g[:, k] - r_u, Qrg)  # reference governor cost
        if k > 0:
            objective += quad_form(g[:, k] - g[:, k - 1], QDg)
            objective += quad_form(y[:, k] - y[:, k - 1], QDy)
        else:
            objective += quad_form(g[:, k] - gminus1, QDg)  # ... penalize the variation of u0 with respect to uold
            objective += quad_form(y[:, k] - yminus1, QDy)  # ... penalize the variation of u0 with respect to uold

        if k < Np - 1:
            constraints += [x[:, k+1] == Ad @x[:, k] + Bd @ g[:, k]]  # system dynamics constraint
        constraints += [ymin <= y[:, k], y[:, k] <= ymax]  # state interval constraint
        constraints += [gmin <= g[:, k], g[:, k] <= gmax]  # input interval constraint

        # if k > 0:
        #     constraints += [Dgmin <= g[:, k] - g[:, k - 1], g[:, k] - g[:, k - 1] <= Dgmax]
        # else:
        #     constraints += [Dgmin <= g[:, k] - gminus1, g[:, k] - gminus1 <= Dgmax]


    prob = Problem(Minimize(objective), constraints)

    # Simulate in closed loop
    nsim = int(len_sim/Ts)  # simulation length(timesteps)
    xsim = np.zeros((nsim, nx))
    ysim = np.zeros((nsim, ny))
    ysim_final = np.zeros((nsim, ny))
    ysim_final[0, :] = np.array([100, 100, 100, 100])


    gsim = np.zeros((nsim, ng))
    tsol = np.zeros((nsim, 1))
    tsim = np.arange(0, nsim)*Ts

    gMPC = ginit  # initial previous measured input is the input at time instant -1.
    time_start = time.time()

    # Basic Kalman filter design
    Q_kal = 10 * np.eye(nx)
    R_kal = np.eye(ny)
    L, P, W = kalman_design_simple(Ad, Bd, Cd, Dd, Q_kal, R_kal, type='filter')
    std_npos = 0 * 0.005

    r_y.value = np.array([200, 100, 200, 200])
    ysim[0, :] = np.array([0, 0, 0, 0])

    # 设置控制输入和控制输出参考值
    r_u.value = np.array([50, 50])  # Reference output

    for i in range(nsim):
        y0 = y0 - r_y.value
        if i==0:
            x0 = x0 + L @ y0
            x0 = Ad.dot(x0) + Bd.dot(gMPC)
            yold = Cd @ x0 + Dd @ gMPC
        else:
            # kalman滤波估计当前状态值，y1是测量值（估计值加上随机数）
            # y0 = yold + std_npos * np.random.randn()
            # y0 = yold
            x0 = x0 + L @ (y0 - yold)
            # 下一步状态值和y估计值
            x0 = Ad.dot(x0) + Bd.dot(gMPC)
            yold = Cd @ x0 + Dd @ gMPC


        # 测试
        if i < 99:
            ysim[i+1, :] = yold
        # print('x0:', x0)
        # print('gMPC:', gMPC)
        # print('yold', yold)

        x_init.value = x0  # set value to the x_init cvx parameter to x0
        gminus1.value = gMPC
        yminus1.value = yold

        time_start = time.time()
        prob.solve(solver=OSQP, warm_start=True)
        tsol[i] = 1000*(time.time() - time_start)

        gMPC = g[:, 0].value
        gsim[i, :] = gMPC
        xsim[i, :] = x0

        if i < 99:
            ysim_final[i + 1, :] = Cd @ x0 + Dd @ gMPC + r_y.value
            y0 = ysim_final[i + 1, :]

    time_sim = time.time() - time_start

    # In[Plot time traces]
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    axes[0].plot(tsim, ysim_final[:, 0], "k", label='p0')
    axes[0].set_title("Output (-)")

    axes[1].plot(tsim, ysim_final[:, 1], "k", label='p1')
    axes[1].set_title("Output (-)")

    axes[2].plot(tsim, gsim[:, 0], label="u0")
    axes[2].plot(tsim, gsim[:, 1], label="u1")
    axes[2].set_title("Input (N)")

    for ax in axes:
        ax.grid(True)
        ax.legend()

    # In[Timing]
    plt.figure()
    # plt.hist(tsol[1:])
    # plt.xlabel("MPC solution time (ms)")
    plt.show()

    print(f"First MPC execution takes {tsol[0, 0]:.0f} ms")