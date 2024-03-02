import numpy as np


class Fem1dEigen:
    """１次元固有値問題 (d^2/d^2x + V)u = λu を解く"""

    def __init__(self, xs, v, dirichlet_bc):
        # Parameters for K matrix
        self.size = len(v)  # Using the same size as the previous matrix for consistency
        self.dirichlet_bc = dirichlet_bc
        self.xs = xs
        self.xs_next = np.roll(xs, -1)  # 次の座標点
        self.element_lengths = self.xs_next - xs  # 要素の長さ
        self.element_lengths = self.element_lengths[:-1]
        self.hl = 1 / 100
        self.v = v

    def get_K(self):
        # Initialize the K matrix with zeros
        K = np.zeros((self.size - 1, self.size - 1))

        # Populate the K matrix
        for i in range(self.size - 1):
            if i == 0:
                K[i, i] = 2 - self.dirichlet_bc[0]
                if self.size > 1:  # To handle the case when size is 1
                    K[i, i + 1] = -1
            elif i == self.size - 2:
                K[i, i - 1] = -1
                K[i, i] = 2 - self.dirichlet_bc[1]
            else:
                K[i, i - 1] = -1
                K[i, i] = 2
                K[i, i + 1] = -1

        # Apply the scale factor
        K *= 1 / self.hl
        return K

    def get_M(self):
        M = np.zeros((self.size - 1, self.size - 1))

        for i in range(self.size - 1):
            if i == 0:
                M[i, i] = 4 + self.dirichlet_bc[0]
                if self.size > 1:  # To handle the case when size is 1
                    M[i, i + 1] = 1
            elif i == self.size - 2:
                M[i, i - 1] = 1
                M[i, i] = 4 + self.dirichlet_bc[1]
            else:
                M[i, i - 1] = 1
                M[i, i] = 4
                M[i, i + 1] = 1

        # Apply the scale factor
        M *= self.hl / 6
    
    def get_V(self):
