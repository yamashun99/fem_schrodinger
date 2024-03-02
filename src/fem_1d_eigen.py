import numpy as np


class Fem1dEigen:
    """１次元固有値問題 (K + V)u = λu を解く"""

    def __init__(self, n, a, b, k, v):
        # Parameters for K matrix
        hl = 1  # Example value for h^l
        size = len(v)  # Using the same size as the previous matrix for consistency

        # Initialize the K matrix with zeros
        K = np.zeros((size, size))

        # Populate the K matrix
        for i in range(size):
            if i == 0:
                K[i, i] = 1
                if size > 1:  # To handle the case when size is 1
                    K[i, i + 1] = -1
            elif i == size - 1:
                K[i, i - 1] = -1
                K[i, i] = 1
            else:
                K[i, i - 1] = -1
                K[i, i] = 2
                K[i, i + 1] = -1

        # Apply the scale factor
        K *= 1 / hl

        K
