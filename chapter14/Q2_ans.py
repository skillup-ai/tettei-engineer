import numpy as np

class RNN:
    def __init__(self, Wx, Wh, h0):
        """
        Wx : 入力xにかかる重み（1,隠れ層のノード数）
        Wh : 1時刻前のhにかかる重み（隠れ層のノード数, 隠れ層のノード数）
        h0 : 隠れ層の初期値（1,隠れ層のノード数）
        """

        # パラメータのリスト
        self.params = [Wx, Wh]

        # 隠れ層の初期値を設定
        self.h_prev = h0

    def forward(self, x):
        """
        順伝播計算
        x : 入力（データ数,1）
        """
        Wx, Wh = self.params
        h_prev = self.h_prev

        t = np.dot(h_prev, Wh) + np.dot(x, Wx)

        # 活性化関数は恒等写像関数とする
        h_next = t

        # 隠れ層の状態の保存
        self.h_prev = h_next

        return h_next

    
if __name__=="__main__":
    Wx = np.arange(1*5).reshape(1, 5)
    print("Wx\n", Wx)
    Wh = np.arange(5*5).reshape(5, 5)
    print("Wh\n", Wh)    
    h0 = np.arange(5).reshape(1, 5)
    print("h0\n", h0) 
    
    rnn = RNN(Wx, Wh, h0)
    
    x = np.arange(3*1).reshape(3, 1)
    print("x\n", x)
    
    h_next = rnn.forward(x)
    print("h_next\n", h_next)
