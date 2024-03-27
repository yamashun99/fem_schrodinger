import numpy as np


class Mel1d:
    """
    1次元要素行列を生成するクラス
    K_ij: \int_0^L dx dV_i/dx c(x) d\phi_j/dx
    M_ij: \int_0^L dx \alpha(x) V_i \phi_j
    V_i, \phi_i はハット関数
    """

    def __init__(self, xs):
        # Parameters for K matrix
        self.size = len(xs)
        self.xs = xs
        self.xs_next = np.roll(xs, -1)  # 次の座標点
        self.hs = self.xs_next - xs  # 要素の長さ
        self.hs = self.hs[:-1]

    def get_K(self, c=None):
        if c is None:
            c = np.ones(self.size - 1)
        # Initialize the K matrix with zeros
        K = np.zeros((self.size, self.size))

        # Populate the K matrix
        for i in range(self.size):
            if i == 0:
                K[i, i] = -1 / self.hs[i] * c[i]
                K[i, i + 1] = 1 / self.hs[i] * c[i]
            elif i == self.size - 1:
                K[i, i] = -1 / self.hs[i - 1] * c[i - 1]
                K[i, i - 1] = 1 / self.hs[i - 1] * c[i - 1]
            else:
                K[i, i - 1] = 1 / self.hs[i - 1] * c[i - 1]
                K[i, i] = -c[i - 1] / self.hs[i - 1] - c[i] / self.hs[i]
                K[i, i + 1] = c[i] / self.hs[i]
        return K

    def get_M(self, alpha=None):
        if alpha is None:
            alpha = np.ones(self.size - 1)
        M = np.zeros((self.size, self.size))
        for i in range(self.size):
            if i == 0:
                M[i, i] = self.hs[i] / 3 * alpha[i]
                M[i, i + 1] = self.hs[i] / 6 * alpha[i]
            elif i == self.size - 1:
                M[i, i - 1] = self.hs[i - 1] / 6 * alpha[i - 1]
                M[i, i] = self.hs[i - 1] / 3 * alpha[i - 1]
            else:
                M[i, i - 1] = self.hs[i] / 6 * alpha[i - 1]
                M[i, i] = self.hs[i] / 3 * (alpha[i - 1] + alpha[i])
                M[i, i + 1] = self.hs[i] / 6 * alpha[i]
        return M
