import numpy as np


class Mel2d:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.n = len(x)
        self.m = len(y)

    def generate_mesh(self):
        # n と m は x と y 方向の分割数
        # p 行列の生成
        X, Y = np.meshgrid(self.x, self.y)
        p = np.vstack((X.flatten(), Y.flatten())).T

        # t 行列の生成
        t = []
        for i in range(self.m - 1):
            for j in range(self.n - 1):
                # 各正方形を二つの三角形に分割
                t.append([i * self.n + j, i * self.n + j + 1, (i + 1) * self.n + j])
                t.append(
                    [i * self.n + j + 1, (i + 1) * self.n + j + 1, (i + 1) * self.n + j]
                )
        t = np.array(t)

        b = []
        for i in range(self.m):
            for j in range(self.n):
                # 辺の点を識別
                if i == 0 or i == self.m - 1 or j == 0 or j == self.n - 1:
                    b.append(i * self.n + j)

        return p, t, b

    def stiffness_matrix(self):
        p, t, b = self.generate_mesh()
        K = np.zeros((self.n * self.m, self.n * self.m))
        for e in range(len(t)):
            Pe = np.hstack((np.ones((3, 1)), p[t[e]]))
            C = np.linalg.inv(Pe)
            grad = C[1:3]
            Area = 0.5 * np.abs(np.linalg.det(Pe))
            Ke = Area * np.dot(grad.T, grad)
            for i in range(3):
                for j in range(3):
                    K[t[e][i], t[e][j]] += Ke[i, j]
        return K

    def force(self):
        p, t, b = self.generate_mesh()
        f = np.zeros(self.n * self.m)
        for e in range(len(t)):
            Pe = np.hstack((np.ones((3, 1)), p[t[e]]))
            Area = 0.5 * np.abs(np.linalg.det(Pe))
            fe = Area / 3 * np.ones(3)
            for i in range(3):
                f[t[e][i]] += fe[i]
        return f

    def boundary_condition(self):
        p, t, b = self.generate_mesh()
        K = self.stiffness_matrix()
        for i in b:
            K[i] = 0
            K[i, i] = 1
        return K
