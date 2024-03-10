import numpy as np


class Fem1dEigen:
    """１次元固有値問題 (d^2/d^2x + V)u = λu を解く"""

    def __init__(self, xs, v):
        # Parameters for K matrix
        self.size = len(v)  # Using the same size as the previous matrix for consistency
        self.xs = xs
        self.xs_next = np.roll(xs, -1)  # 次の座標点
        self.hs = self.xs_next - xs  # 要素の長さ
        self.hs = self.hs[:-1]
        self.v = v

    def get_K(self):
        # Initialize the K matrix with zeros
        K = np.zeros((self.size, self.size))

        # Populate the K matrix
        for i in range(self.size):
            if i == 0:
                K[i, i] = 1 / self.hs[i]
                K[i, i + 1] = -1 / self.hs[i]
            elif i == self.size - 1:
                K[i, i] = 1 / self.hs[i - 1]
                K[i, i - 1] = -1 / self.hs[i - 1]
            else:
                K[i, i - 1] = -1 / self.hs[i - 1]
                K[i, i] = 1 / self.hs[i - 1] + 1 / self.hs[i]
                K[i, i + 1] = -1 / self.hs[i]
        return K

    def get_M(self):
        M = np.zeros((self.size, self.size))

        for i in range(self.size):
            if i == 0:
                M[i, i] = 2 * self.hs[i]
                M[i, i + 1] = self.hs[i]
            elif i == self.size - 1:
                M[i, i - 1] = self.hs[i - 1]
                M[i, i] = 2 * self.hs[i - 1]
            else:
                M[i, i - 1] = self.hs[i - 1]
                M[i, i] = 2 * self.hs[i - 1] + 2 * self.hs[i]
                M[i, i + 1] = self.hs[i]

        return M / 6

    def get_V(self):
        V = np.zeros((self.size, self.size))
        for i in range(self.size):
            if i == 0:
                V[i, i] = (3 * self.v[i] + self.v[i + 1]) * self.hs[i]
                V[i, i + 1] = (self.v[i] + self.v[i + 1]) * self.hs[i]
            elif i == self.size - 1:
                V[i, i] = (self.v[i - 1] + 3 * self.v[i]) * self.hs[i - 1]
                V[i, i - 1] = (self.v[i - 1] + self.v[i]) * self.hs[i - 1]
            else:
                V[i, i] = (3 * self.v[i - 1] + self.v[i]) * self.hs[i - 1] + (
                    self.v[i] + 3 * self.v[i + 1]
                ) * self.hs[i]
                V[i, i - 1] = (self.v[i - 1] + self.v[i]) * self.hs[i - 1]
                V[i, i + 1] = (self.v[i] + self.v[i + 1]) * self.hs[i]
        return V / 12
