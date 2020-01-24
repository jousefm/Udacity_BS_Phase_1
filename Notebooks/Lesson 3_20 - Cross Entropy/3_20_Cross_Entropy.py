import numpy as np


# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    return -1 * np.sum(np.matmul(Y, np.log(P)) + np.matmul((1 - Y), np.log(1 - P)))


if __name__ == "__main__":
    Y = np.array([1, 1, 0])
    P = np.array([0.8, 0.7, 0.1])

    assert round(cross_entropy(Y, P), 2) == 0.69  # low cross entropy

    Y = np.array([1, 0, 1, 1])
    P = np.array([0.4, 0.6, 0.1, 0.5])

    assert round(cross_entropy(Y, P), 2) == 4.83  # high cross entropy due to vectors being different