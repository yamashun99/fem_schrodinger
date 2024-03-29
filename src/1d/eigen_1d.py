import numpy as np


class Fem1dEigen:
    """１次元固有値問題 (d^2/d^2x + V)u = λu を解く"""

    def __init__(self, xs):
        # Parameters for K matrix
        self.size = len(xs)
        self.xs = xs
        self.xs_next = np.roll(xs, -1)  # 次の座標点
        self.hs = self.xs_next - xs  # 要素の長さ
        self.hs = self.hs[:-1]

    def get_K0(self):
        # Initialize the K matrix with zeros
        K = np.zeros((self.size, self.size))

        # Populate the K matrix
        for i in range(self.size):
            if i == 0:
                K[i, i] = -1 / self.hs[i]
                K[i, i + 1] = 1 / self.hs[i]
            elif i == self.size - 1:
                K[i, i] = -1 / self.hs[i - 1]
                K[i, i - 1] = 1 / self.hs[i - 1]
            else:
                K[i, i - 1] = 1 / self.hs[i - 1]
                K[i, i] = -1 / self.hs[i - 1] - 1 / self.hs[i]
                K[i, i + 1] = 1 / self.hs[i]
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

    def get_K1(self, v):
        K = np.zeros((self.size, self.size))
        for i in range(self.size):
            if i == 0:
                K[i, i] = self.hs[i] / 3 * v[i]
                K[i, i + 1] = self.hs[i] / 6 * v[i]
            elif i == self.size - 1:
                K[i, i - 1] = self.hs[i - 1] / 6 * v[i - 1]
                K[i, i] = self.hs[i - 1] / 3 * v[i - 1]
            else:
                K[i, i - 1] = self.hs[i] / 6 * v[i - 1]
                K[i, i] = self.hs[i] / 3 * (v[i - 1] + v[i])
                K[i, i + 1] = self.hs[i] / 6 * v[i]
        return K
