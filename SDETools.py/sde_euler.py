import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

def sde_euler(drift, diffusion, T, dt, x0, seed=True):
    """
    Simulates the Euler-Maruyama method for a stochastic differential equation.

    Args:
        drift: The drift function of the SDE.
        diffusion: The diffusion function of the SDE.
        T: The final time.
        dt: The time step.
        x0: The initial value.

    Returns:
        A NumPy array of the simulated values.
    """

    if seed:
        np.random.seed(1)

    x = np.zeros((int(T / dt) + 1, x0.size))
    x[0] = x0
    for i in range(1, len(x)):
        dW = st.norm(0, np.sqrt(dt)).rvs(size=x0.size)
        x[i] = x[i - 1] + drift(x[i - 1]) * dt + diffusion(x[i - 1]) * dW

    return x