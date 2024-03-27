import numpy as np
from scipy.sparse import coo_matrix


class Mel1dSparse:
    def __init__(self, xs):
        self.size = len(xs)
        self.xs = xs
        self.xs_next = np.roll(xs, -1)
        self.hs = self.xs_next - xs
        self.hs = self.hs[:-1]

    def get_K(self, c=None, bc=[None, None]):
        if c is None:
            c = np.ones(self.size - 1)

        row = []
        col = []
        data = []

        for i in range(self.size):
            if i == 0:
                row.extend([i, i])
                col.extend([i, i + 1])
                data.extend([-1 / self.hs[i] * c[i], 1 / self.hs[i] * c[i]])
            elif i == self.size - 1:
                row.extend([i, i])
                col.extend([i, i - 1])
                data.extend(
                    [-1 / self.hs[i - 1] * c[i - 1], 1 / self.hs[i - 1] * c[i - 1]]
                )
            else:
                row.extend([i, i, i])
                col.extend([i - 1, i, i + 1])
                data.extend(
                    [
                        1 / self.hs[i - 1] * c[i - 1],
                        -c[i - 1] / self.hs[i - 1] - c[i] / self.hs[i],
                        c[i] / self.hs[i],
                    ]
                )

        K_sparse = coo_matrix((data, (row, col)), shape=(self.size, self.size))
        return K_sparse

    def get_M(self, alpha=None):
        if alpha is None:
            alpha = np.ones(self.size - 1)

        row = []
        col = []
        data = []

        for i in range(self.size):
            if i == 0:
                row.extend([i, i])
                col.extend([i, i + 1])
                data.extend([self.hs[i] / 3 * alpha[i], self.hs[i] / 6 * alpha[i]])
            elif i == self.size - 1:
                row.extend([i, i])
                col.extend([i - 1, i])
                data.extend(
                    [
                        self.hs[i - 1] / 6 * alpha[i - 1],
                        self.hs[i - 1] / 3 * alpha[i - 1],
                    ]
                )
            else:
                row.extend([i, i, i])
                col.extend([i - 1, i, i + 1])
                data.extend(
                    [
                        self.hs[i] / 6 * alpha[i - 1],
                        self.hs[i] / 3 * (alpha[i - 1] + alpha[i]),
                        self.hs[i] / 6 * alpha[i],
                    ]
                )

        M_sparse = coo_matrix((data, (row, col)), shape=(self.size, self.size))
        return M_sparse

    def get_F(self, f=None):
        if f is None:
            f = np.ones(self.size - 1)

        F = np.zeros(self.size)

        for i in range(self.size):
            if i == 0:
                F[i] = self.hs[i] / 2 * f[i]
            elif i == self.size - 1:
                F[i] = self.hs[i - 1] / 2 * f[i - 1]
            else:
                F[i] = self.hs[i - 1] / 2 * f[i - 1] + self.hs[i] / 2 * f[i]

        return F
