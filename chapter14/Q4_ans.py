import numpy as np

def forward(x, h_prev, c_prev, Wx, Wh, b):
    """
    順伝播計算
    x: 入力 (データ数, 特徴量の数)
    h_prev: 前時刻の隠れ層の出力 （データ数, 隠れ層のノード数）
    c_prev: 前時刻のメモリの状態 （データ数, 隠れ層のノード数）
    Wx: 入力x用の重みパラメータ （特徴量の数, 4×隠れ層のノード数）
    Wh: 隠れ状態h用の重みパラメータ（隠れ層のノード数, 4×隠れ層のノード数）
    b: バイアス （4×隠れ層のノード数）
    """

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

    return h_next, c_next


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
    
if __name__=="__main__":
    N = 2
    D = 3
    H = 4
    x = np.random.randn(N, D)
    print("x\n", x)
    
    h_prev = np.random.randn(N, H)
    print("h_prev\n", h_prev)
    
    c_prev = np.random.randn(N, H)
    print("c_prev\n", c_prev)
    
    Wx = np.random.randn(D, 4*H)
    print("Wx\n", Wx)
    
    Wh = np.random.randn(H, 4*H)
    print("Wh\n", Wh)   
    
    b = np.random.randn(4*H)
    print("b\n", b) 
    
    h_next, c_next = forward(x, h_prev, c_prev, Wx, Wh, b)
    print("h_next\n", h_next)
    print("c_next\n", c_next)    