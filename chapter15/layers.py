import numpy as np

def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))
    return x


class Embedding:
    def __init__(self, W):
        """
        W : 重み行列, word2vecの埋め込み行列に相当する。配列形状は、(語彙数、埋め込みベクトルの要素数)
        """
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        """
        順伝播計算
        """
        W, = self.params # dWの後の,はリストから1つだけを抜き出すためにつけている
        self.idx = idx
        
        # 埋め込み行列から埋め込みベクトルを取り出す
        out = W[idx]
        
        return out
    
    
class TimeEmbedding:
    def __init__(self, W):
        """
        W : 重み行列, word2vecの埋め込み行列に相当する。配列形状は、(語彙数、埋め込みベクトルの要素数)
        """        
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None
        self.W = W

    def forward(self, xs):
        """
        順伝播計算
        xs : 入力の単語ID, 配列形状は(バッチサイズ、時間数)
        """
        N, T = xs.shape # バッチサイズ、時間数
        V, D = self.W.shape # 語彙数、埋め込みベクトルの要素数

        # 初期化
        out = np.empty((N, T, D), dtype='f')
        self.layers = []

        # 時間方向に計算を進める
        for t in range(T):
            
            # Embeddigレイヤを生成し、順伝播計算を行う
            layer = Embedding(self.W)
            out[:, t, :] = layer.forward(xs[:, t])
            
            #  Embeddigレイヤを保持しておく
            self.layers.append(layer)

        return out
    
    
class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1

    def forward(self, xs, ts):
        N, T, V = xs.shape

        if ts.ndim == 3:  # 教師ラベルがone-hotベクトルの場合
            ts = ts.argmax(axis=2)

        mask = (ts != self.ignore_label)

        # バッチ分と時系列分をまとめる（reshape）
        xs = xs.reshape(N * T, V)
        ts = ts.reshape(N * T)
        mask = mask.reshape(N * T)

        ys = softmax(xs)
        ls = np.log(ys[np.arange(N * T), ts])
        ls *= mask  # ignore_labelに該当するデータは損失を0にする
        loss = -np.sum(ls)
        loss /= mask.sum()

        self.cache = (ts, ys, mask, (N, T, V))
        return loss    