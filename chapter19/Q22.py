import numpy as np
from my_functions import sigmoid #シグモイド関数

class GRU:
    def __init__(self, Wx, Wh, b):
        '''
        Wx: 入力x用の重みパラメータ（3つ分の重みをまとめたもの）
        Wh: 隠れ状態h用の重みパラメータ（3つ分の重みをまとめたもの）
        b: バイアス（3つ分のバイアスをまとめたもの）
        '''
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev):
        """
        順伝播計算
        """
        Wx, Wh, b = self.params
        N, H = h_prev.shape
        
        Wxz, Wxr, Wxh = Wx[:, :H], Wx[:, H:2 * H], Wx[:, 2 * H:]
        Whz, Whr, Whh = Wh[:, :H], Wh[:, H:2 * H], Wh[:, 2 * H:]
        bhz, bhr, bhh =  b[:H], b[H:2 * H], b[2 * H:]
        
        z = sigmoid(np.dot(x, Wxz) + np.dot(h_prev, Whz) + bhz)
        r = sigmoid(np.dot(x, Wxr) + np.dot(h_prev, Whr) + bhr)
        h_hat = np.tanh(np.dot(x, Wxh) + np.dot(r * h_prev, Whh) + bhh)
        h_next = z * h_prev + (1 - z) * h_hat

        self.cache = (x, h_prev, z, r, h_hat)

        return h_next

    def backward(self, dh_next):
        """
        逆伝播計算（省略）
        """       
        return dx, dh_prev