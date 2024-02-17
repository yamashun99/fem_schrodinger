import numpy as np


class fem_shrodinger:
    def __init__(self, L, V):
        self.N = len(V)
        self.L = L
        self.V = V
        self.le = L / self.N
        self.hbar = 1.0545718e-34
        self.m_e = 9.10938356e-31

    def get_Aes(
        self,
    ):
        Aes = np.zeros((self.N, 2, 2))
        Aes[:, 0, 0] = (
            self.hbar**2 / (2 * self.m_e)
            + 1 / 12 * (3 * self.V + np.roll(self.V, -1)) * self.le
        )
        Aes[:, 0, 1] = (
            -(self.hbar**2) / (2 * self.m_e)
            + 1 / 12 * (self.V + np.roll(self.V, -1)) * self.le
        )
        Aes[:, 1, 0] = (
            -(self.hbar**2) / (2 * self.m_e)
            + 1 / 12 * (self.V + np.roll(self.V, -1)) * self.le
        )
        Aes[:, 1, 1] = (
            self.hbar**2 / (2 * self.m_e)
            + 1 / 12 * (self.V + 3 * np.roll(self.V, -1)) * self.le
        )
        return Aes

    def get_Be(self):
        Be = np.array(
            [
                [self.le / 3, self.le / 6],
                [self.le / 6, self.le / 3],
            ]
        )
        return Be

    def get_A(self):
        A = np.zeros((self.N + 1, self.N + 1))
        Aes = self.get_Aes()
        for i in range(self.N):
            A[i : i + 2, i : i + 2] += Aes[i]
        return A

    def get_B(self):
        B = np.zeros((self.N + 1, self.N + 1))
        Be = self.get_Be()
        for i in range(self.N):
            B[i : i + 2, i : i + 2] += Be
        return B


class FemSchrodinger:
    def __init__(self, length, potential):
        """シュレーディンガー方程式のFEM解析クラスの初期化
        Args:
            length (float): 系の長さ
            potential (numpy.ndarray): 離散化されたポテンシャル配列
        """
        self.N = len(potential)  # 離散化点の数
        self.length = length
        self.potential = potential
        self.element_length = length / self.N  # 各要素の長さ
        self.hbar = 1.0545718e-34  # 換算プランク定数
        self.m_e = 9.10938356e-31  # 電子質量
        # self.hbar = 1  # 換算プランク定数
        # self.m_e = 1  # 電子質量

    def get_element_matrices(self):
        """要素行列Aesを計算"""
        potential_next = np.roll(self.potential, -1)  # ポテンシャル配列を1つシフト
        coef = self.hbar**2 / (2 * self.m_e) / self.element_length  # 計算に頻出する係数
        # coef = self.hbar**2 / (2 * self.m_e)  # 計算に頻出する係数
        Aes = np.zeros((self.N, 2, 2))
        Aes[:, 0, 0] = (
            coef + 1 / 12 * (3 * self.potential + potential_next) * self.element_length
        )
        Aes[:, 0, 1] = Aes[:, 1, 0] = (
            -coef + 1 / 12 * (self.potential + potential_next) * self.element_length
        )
        Aes[:, 1, 1] = (
            coef + 1 / 12 * (self.potential + 3 * potential_next) * self.element_length
        )
        return Aes

    def get_element_matrix_B(self):
        """一定の要素行列Beを計算"""
        return np.array(
            [
                [self.element_length / 3, self.element_length / 6],
                [self.element_length / 6, self.element_length / 3],
            ]
        )

    def assemble_matrix_A(self):
        """全体行列Aを組み立て"""
        A = np.zeros((self.N + 1, self.N + 1))
        Aes = self.get_element_matrices()
        for i in range(self.N):
            A[i : i + 2, i : i + 2] += Aes[i]
        return A

    def assemble_matrix_B(self):
        """全体行列Bを組み立て"""
        B = np.zeros((self.N + 1, self.N + 1))
        Be = self.get_element_matrix_B()
        for i in range(self.N):
            B[i : i + 2, i : i + 2] += Be
        return B
