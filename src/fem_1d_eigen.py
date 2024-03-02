import numpy as np


class Fem1dEigen:
    """１次元固有値問題 (d^2/d^2x + V)u = λu を解く"""

    def __init__(self, v):
        # Parameters for K matrix
        self.hl = 1  # Example value for h^l
        self.size = len(v)  # Using the same size as the previous matrix for consistency

    def get_K(self):
        # Initialize the K matrix with zeros
        K = np.zeros((self.size, self.size))

        # Populate the K matrix
        for i in range(self.size):
            if i == 0:
                K[i, i] = 1
                if self.size > 1:  # To handle the case when size is 1
                    K[i, i + 1] = -1
            elif i == self.size - 1:
                K[i, i - 1] = -1
                K[i, i] = 1
            else:
                K[i, i - 1] = -1
                K[i, i] = 2
                K[i, i + 1] = -1

        # Apply the scale factor
        K *= 1 / self.hl
        return K

    def get_M(self):
        M = np.zeros((self.size, self.size))

        for i in range(self.size):
            if i == 0:
                M[i, i] = 2
                if self.size > 1:  # To handle the case when size is 1
                    M[i, i + 1] = 1
            elif i == self.size - 1:
                M[i, i - 1] = 1
                M[i, i] = 2
            else:
                M[i, i - 1] = 1
                M[i, i] = 4
                M[i, i + 1] = 1

        # Apply the scale factor
        M *= 1 / self.hl
        return M
