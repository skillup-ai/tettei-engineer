import numpy as np

def forward(self, x, h_prev, c_prev):
    """
    順伝播計算
    x: 入力
    h_prev: 前時刻の隠れ層の出力
    c_prev: 前時刻のメモリの状態
    Wx: 入力x用の重みパラメータ（4つ分の重みをまとめたもの）
    Wh: 隠れ状態h用の重みパラメータ（4つ分の重みをまとめたもの）
    b: バイアス（4つ分のバイアスをまとめたもの）
    """
    Wx, Wh, b = self.params
    N, H = h_prev.shape

    A = np.dot(x, Wx) + np.dot(h_prev, Wh) + b

    f = A[:, :H]
    g = A[:, H:2*H]
    i = A[:, 2*H:3*H]
    o = A[:, 3*H:]

    f = sigmoid(f)
    g = np.tanh(g)
    i = sigmoid(i)
    o = sigmoid(o)

    print(f.shape, c_prev.shape, g.shape, i.shape)
    c_next = f * c_prev + g * i
    h_next = o * np.tanh (c_next)

    self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)
    return h_next, c_next
