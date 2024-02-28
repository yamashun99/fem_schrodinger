import numpy as np
from scipy.constants import hbar, m_e, epsilon_0, e  # 物理定数をインポート


class FemRadial:
    def __init__(self, xs):
        """ラジアルシュレーディンガー方程式のFEM解析クラスの初期化
        Args:
            xs (numpy.ndarray): 系の離散化された座標点配列
        """
        if len(xs) < 2:
            raise ValueError("xsは少なくとも2つの要素を含む必要があります。")
        self.length = xs[-1] - xs[0]  # 系の全長
        self.xs = xs
        self.xs_next = np.roll(self.xs, -1)  # 次の座標点
        self.element_lengths = self.xs_next - self.xs  # 要素の長さ
        self.element_matrix = np.zeros((len(xs), 2, 2))
        self.element_matrix[:, 0, 0] = self.element_matrix[:, 0, 1] = (
            self.element_matrix[:, 1, 0]
        ) = self.element_matrix[:, 1, 1] = self.element_lengths
        self.N = len(xs)
        self.C0 = -(hbar**2) / (2 * m_e)
        self.C1 = -(e**2) / (4 * np.pi * epsilon_0)

    def get_block_r2drfdrf(self):
        """運動エネルギーのブロック行列を取得"""
        K = np.zeros((self.N, 2, 2))
        K[:, 0, 0] = K[:, 1, 1] = self.xs**2 + self.xs * self.xs_next + self.xs_next**2
        K[:, 1, 0] = K[:, 0, 1] = -(
            self.xs**2 + self.xs * self.xs_next + self.xs_next**2
        )
        return K / 3 / self.element_matrix

    def get_block_rff(self):
        """ポテンシャルエネルギーのブロック行列を取得"""
        V = np.zeros((self.N, 2, 2))
        V[:, 0, 0] = 3 * self.xs + self.xs_next
        V[:, 0, 1] = V[:, 1, 0] = self.xs + self.xs_next
        V[:, 1, 1] = self.xs + 3 * self.xs_next
        return V * self.element_matrix / 12

    def get_block_r2ff(self):
        M = np.zeros((self.N, 2, 2))
        M[:, 0, 0] = (
            self.element_lengths
            * (6 * self.xs**2 + 3 * self.xs * self.xs_next + self.xs_next**2)
            / 30
        )
        M[:, 0, 1] = M[:, 1, 0] = (
            (
                -3 * self.xs**5
                + 5 * self.xs**4 * self.xs_next
                - 5 * self.xs * self.xs_next**4
                + 3 * self.xs_next**5
            )
            / 60
            / self.element_lengths**2
        )
        M[:, 1, 1] = (
            self.element_lengths
            * (self.xs**2 + 3 * self.xs * self.xs_next + 6 * self.xs_next**2)
            / 30
        )
        return M

    def assemble_matrix(self, block):
        """ブロックから全体の行列を組み立てる"""
        A = np.zeros((self.N + 1, self.N + 1))
        for i in range(self.N):
            A[i : i + 2, i : i + 2] += block[i]
        return A

    def get_K(self):
        """運動エネルギー行列を取得"""
        return -self.C0 * self.assemble_matrix(self.get_block_r2drfdrf())

    def get_V(self):
        """ポテンシャルエネルギー行列を取得"""
        return self.C1 * self.assemble_matrix(self.get_block_rff())

    def get_M(self):
        return self.assemble_matrix(self.get_block_r2ff())

    def get_V_poisson(self):
        return -self.assemble_matrix(self.get_block_r2drfdrf())

    def get_source_poisson(self):
        return -(e**2) / epsilon_0 * self.assemble_matrix(self.get_block_r2ff())
