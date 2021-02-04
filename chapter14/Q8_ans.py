import numpy as np

def forward(hs, h):
    """
    順伝播
    重みベクトルを求めるための関数
    hs : エンコーダにおけるすべての隠れ状態（データ数,時刻数,隠れ層のノード数）
    h : デコーダにおける、ある時刻の隠れ状態（データ数,隠れ層のノード数）
    """
    N, T, H = hs.shape

    #　デコーダのある場所の隠れ状態を3次元配列に変形する
    hr = h.reshape(N, 1, H).repeat(T, axis=1)

    # エンコーダの隠れ状態とコーダの隠れ状態を掛けて足し合わせることで内積をとる
    # ほかの実装例として、hsとhrを結合し、重みWを掛けるという方法もある
    t = hs * hr
    s = np.sum(t, axis=2)

    # ソフトマックス関数に通すことで、正規化する
    a = softmax(s) # aは重みベクトルを並べた行列 (N * T)

    return a


def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))
    return x


if __name__=="__main__":
    N = 2
    T = 3
    H = 4
    hs = np.random.randn(N, T, H)
    print("hs\n", hs)
    
    h = np.random.randn(N, H)
    print("h\n", h)
    
    a = forward(hs, h)
    print("a\n", a)