from typing import Tuple

import numpy as np
from scipy.special import expit  # Sigmoid function


def time_mixing(
    x: np.ndarray,
    last_x: np.ndarray,
    last_num: np.ndarray,
    last_den: np.ndarray,
    decay: np.ndarray,
    bonus: np.ndarray,
    mix_k: np.ndarray,
    mix_v: np.ndarray,
    mix_r: np.ndarray,
    Wk: np.ndarray,
    Wv: np.ndarray,
    Wr: np.ndarray,
    Wout: np.ndarray,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Perform time mixing operation using the RWKV model.

    Args:
        x (np.ndarray): Current time step embedding
        last_x (np.ndarray): Previous time step embedding
        last_num (np.ndarray): Numerator, or "weighted sum of past values" from the last time step
        last_den (np.ndarray): Denominator, "sum of weights of past values" from the last time step
        decay (np.ndarray): Learnable parameter for controlling decay
        bonus (np.ndarray): Learnable parameter for providing a bonus weight to the current value
        mix_k (np.ndarray): Mixing ratio for key
        mix_v (np.ndarray): Mixing ratio for value
        mix_r (np.ndarray): Mixing ratio for receptance
        Wk (np.ndarray): Affine transformation for key (1024, 1024)
        Wv (np.ndarray): Affine transformation for value (1024, 1024)
        Wr (np.ndarray): Affine transformation for receptance (1024, 1024)
        Wout (np.ndarray): Affine transformation for output (1024, 1024)

    Returns:
        Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]: A tuple containing:
            - np.ndarray: Time-mixed output embedding
            - Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the current time step embedding,
              updated numerator, and updated denominator for the next time step.

    Raises:
        ValueError: If input arrays do not have the correct shape.
    """
    # Linear interpolation
    k = Wk @ (x * mix_k + last_x * (1 - mix_k))
    v = Wv @ (x * mix_v + last_x * (1 - mix_v))
    r = Wr @ (x * mix_r + last_x * (1 - mix_r))

    # Numerator and Denominator calculation
    num = np.exp(-np.exp(decay)) * last_num + np.exp(k) * v
    den = np.exp(-np.exp(decay)) * last_den + np.exp(k)

    # Weighted average computation
    wkv = (last_num + np.exp(bonus + k) * v) / (last_den + np.exp(bonus + k))

    # Receptance control
    rwkv = expit(r) * wkv

    # Output calculation
    time_mixed = Wout @ rwkv

    # Return results
    return time_mixed, (x, num, den)
