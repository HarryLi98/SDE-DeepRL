import tensorflow as tf
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

@tf.function
def tf_sde_euler(drift, diffusion, T, dt, x0, seed=True):
    """
    Simulates the Euler-Maruyama method for a stochastic differential equation.

    Args:
        drift: The drift function of the SDE.
        diffusion: The diffusion function of the SDE.
        T: The final time.
        dt: The time step.
        x0: The initial value.

    Returns:
        A TensorFlow array of the simulated values.
    """

    if seed:
        tf.random.set_seed(1)

    x = []
    x.append(x0)
    for i in range(1, (int(T / dt) + 1)):
        dW = tf.random.normal(shape=x0.shape) * tf.sqrt(dt)
        x.append(x[i - 1] + drift(x[i - 1]) * dt + diffusion(x[i - 1]) * dW)

    return tf.stack(x)