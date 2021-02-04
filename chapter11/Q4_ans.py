import numpy as np

def relu(X):
    return np.maximum(0, X)

def softmax(X):
    X = X - np.max(X, axis=1, keepdims=True)
    return np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)

def relu_backward(Z, delta):
    delta[Z == 0] = 0

def cross_entropy_error(y, t):
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


class FullyConnectedNeuralNetwork():

    def __init__(self, layer_units):
        """
        layer_units : list, 各層のノード数を格納したリスト
        """
        self.n_iter_ = 0
        self.t_ = 0
        self.layer_units = layer_units
        self.n_layers_ = len(layer_units)

        # パラメータの初期化
        self.coefs_ = []
        self.intercepts_ = []
        for i in range(self.n_layers_ - 1):
            coef_init, intercept_init = self._init_coef(layer_units[i],layer_units[i+1])
            self.coefs_.append(coef_init)
            self.intercepts_.append(intercept_init)

        # 勾配の初期化
        self.coef_grads_ = [np.empty((n_in_, n_out_)) for n_in_,n_out_
                                             in zip(layer_units[:-1],layer_units[1:])]
        self.intercept_grads_ = [np.empty(n_out_) for n_out_ in layer_units[1:]]


    def _init_coef(self, n_in, n_out):
        """
        ある層間のパラメータを初期化するメソッド
        n_in : int, 入力側のノード数
        n_out : int, 出力側のノード数
        """
        std = np.sqrt(2/n_in)
        coef_init = np.random.randn(n_in, n_out) * std
        intercept_init = np.zeros(n_out)
        return coef_init, intercept_init


    def _forward(self, activations):
        """
        順伝播処理を行うメソッド
        activations : list, 各層の出力を納めたリスト
        　　　　　　　　　　　　　　　　　　　　activations[0]は入力データ
                      　　　　　　　activations[i].shape=(バッチサイズ,ノード数)
        """
        affine = [None] * (self.n_layers_ - 1)
        for i in range(self.n_layers_ - 1):

            # アフィン変換
            affine[i] = np.dot(activations[i], self.coefs_[i]) + self.intercepts_[i]

            if (i + 1) == ( self.n_layers_ - 1 ):
                """
                出力層の場合
                """
                activations[i + 1] = softmax(affine[i])
            else:
                """
                隠れ層の場合
                """
                activations[i + 1] = relu(affine[i])

        return activations


    def _grad(self, j, activations, deltas):
        """
        各パラメータの勾配を算出するメソッド
        j : int, アフィンの番号
        activations : list, 各層の出力を納めたリスト
        deltas : list, 出力層側から伝わってきた勾配を納めたリスト
        """
        self.coef_grads_[j] = np.dot(activations[j].T,deltas[j])
        self.intercept_grads_[j] = np.sum(deltas[j], axis=0)


    def _backward(self, t, activations):
        """
        逆伝播処理を行うメソッド
        t : array-like, 正解ラベル, t.shape=(バッチサイズ, 出力層ノード数)
        activations : list, 各層の出力を納めたリスト
        """
        deltas = [None] * (self.n_layers_ - 1)
        last = self.n_layers_ - 2

        # 交差エントロピー誤差とソフトマックス関数を合わせて勾配を算出
        n_samples = t.shape[0]
        deltas[last] = (activations[-1] - t)/n_samples

        # 出力層の1つ手前のパラメータの勾配を算出
        self._grad(last, activations, deltas)

        # 残りのパラメータの勾配を算出
        for i in range(self.n_layers_ - 2, 0, -1):
            # 入力(activations)の勾配を算出
            deltas[i - 1] = np.dot(deltas[i], self.coefs_[i].T)

            # 活性化関数ReLUの勾配を算出
            relu_backward(activations[i], deltas[i - 1])

            # パラメータの勾配を算出
            self._grad(i-1, activations, deltas)

        return


    def _forward_and_back(self, x, t):
        """
        順伝播処理を実行した後、逆伝播処理を実行するメソッド
        x : array-like, 入力データ, x.shape=(バッチサイズ, 入力層ノード数)
        t : array-like, 正解ラベル, t.shape=(バッチサイズ, 出力層ノード数)
        """
        activations = [x] + [None] * (self.n_layers_ - 1)

        # 順伝播
        activations = self._forward(activations)
        loss = cross_entropy_error(activations[-1], t)

        # 逆伝播
        self._backward(t, activations)

        return loss
