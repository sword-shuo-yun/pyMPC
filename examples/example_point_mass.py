import numpy as np
import scipy.sparse as sparse
import time
import matplotlib.pyplot as plt
from pyMPC.mpc import MPCController
from scipy.integrate import ode


if __name__ == '__main__':
    # Constants #
    Ts = 0.2 # sampling time (s)
    M = 2    # mass (Kg)
    b = 0.3  # friction coefficient (N*s/m)

    # Continous-time matrices (just for reference)
    Ac = np.array([
        [0.0, 1.0],
        [0, -b/M]]
    )
    Bc = np.array([
        [0.0],
        [1/M]
    ])

    def f_ODE(t,x,u):
        der = Ac @ x + Bc @ u
        return der

    [nx, nu] = Bc.shape  # number of states and number or inputs

    # Brutal forward euler discretization
    Ad = np.eye(nx) + Ac*Ts
    Bd = Bc*Ts


    # Reference input and states
    pref = 7.0
    vref = 0.0
    xref = np.array([pref, vref]) # reference state
    uref = np.array([0.0])    # reference input
    uminus1 = np.array([0.0])     # input at time step negative one - used to penalize the first delta u at time instant 0. Could be the same as uref.

    # Constraints
    xmin = np.array([-100.0, -100.0])
    xmax = np.array([100.0,   100.0])

    umin = np.array([-1.2])
    umax = np.array([1.2])

    Dumin = np.array([-2e-1])
    Dumax = np.array([2e-1])

    # Objective function
    Qx = sparse.diags([0.5, 0.1])   # Quadratic cost for states x0, x1, ..., x_N-1
    QxN = sparse.diags([0.5, 0.1])  # Quadratic cost for xN
    Qu = 2.0 * sparse.eye(1)        # Quadratic cost for u0, u1, ...., u_N-1
    QDu = 10.0 * sparse.eye(1)       # Quadratic cost for Du0, Du1, ...., Du_N-1

    # Initial state
    x0 = np.array([0.1, 0.2]) # initial state
    system_dyn = ode(f_ODE).set_integrator('vode', method='bdf')
    system_dyn.set_initial_value(x0, 0)
    system_dyn.set_f_params(0.0)

    # Prediction horizon
    Np = 20

    K = MPCController(Ad,Bd,Np=Np, x0=x0,xref=xref,uminus1=uminus1,
                      Qx=Qx, QxN=QxN, Qu=Qu,QDu=QDu,
                      xmin=xmin,xmax=xmax,umin=umin,umax=umax,Dumin=Dumin,Dumax=Dumax)
    K.setup()

    # Simulate in closed loop
    [nx, nu] = Bd.shape # number of states and number or inputs
    len_sim = 15 # simulation length (s)
    nsim = int(len_sim/Ts) # simulation length(timesteps)
    xsim = np.zeros((nsim,nx))
    usim = np.zeros((nsim,nu))
    tcalc = np.zeros((nsim,1))
    tsim = np.arange(0,nsim)*Ts



    xstep = x0
    uMPC = uminus1
    for i in range(nsim):
        xsim[i,:] = xstep

        # MPC update and step. Could be in just one function call
        time_start = time.time()
        K.update(xstep, uMPC) # update with measurement
        uMPC = K.output() # MPC step (u_k value)
        tcalc[i,:] = time.time() - time_start
        usim[i,:] = uMPC

        #xstep = Ad.dot(xstep) + Bd.dot(uMPC)  # Real system step (x_k+1 value)
        system_dyn.set_f_params(uMPC) # set current input value to uMPC
        system_dyn.integrate(system_dyn.t + Ts)
        xstep = system_dyn.y


    fig,axes = plt.subplots(3,1, figsize=(10,10))
    axes[0].plot(tsim, xsim[:,0], "k", label='p')
    axes[0].plot(tsim, xref[0]*np.ones(np.shape(tsim)), "r--", label="pref")
    axes[0].set_title("Position (m)")

    axes[1].plot(tsim, xsim[:,1], label="v")
    axes[1].plot(tsim, xref[1]*np.ones(np.shape(tsim)), "r--", label="vref")
    axes[1].set_title("Velocity (m/s)")

    axes[2].plot(tsim, usim[:,0], label="u")
    axes[2].plot(tsim, uref*np.ones(np.shape(tsim)), "r--", label="uref")
    axes[2].set_title("Force (N)")

    for ax in axes:
        ax.grid(True)
        ax.legend()

    plt.figure()
    plt.hist(tcalc*1000)
    plt.grid(True)

