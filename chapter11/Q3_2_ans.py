class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None # ソフトマックス関数の出力を格納するインスタンス変数
        self.t = None # 正解ラベルを格納するインスタンス変数

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self):
        batch_size = self.t.shape[0]
        dx = (self.y-self.t)/batch_size
        return dx

