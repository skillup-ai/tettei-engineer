import numpy as np
from common.functions import softmax


def cross_entropy_error(y, t):
    """
    y : ソフトマックス関数の出力
        y.shape=(k,)またはy.shape=(N,k)
    t : 正解ラベル(ワンホット表現)
        t.shape=(k,)またはt.shape=(N,k)
    """
    if y.ndim==1:
        t = t.reshape(1,-1)
        y = y.reshape(1,-1)

    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(t*np.log(y + delta))/ batch_size


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
    
    
if __name__=="__main__":
    x = np.arange(2*3).reshape(2, 3)
    print("x\n", x)   
    
    t = np.arange(2).reshape(-1, 1)
    print("t\n", t)   
    
    sl = SoftmaxWithLoss()
    
    loss = sl.forward(x,t)    
    print("loss\n", loss)
    
    dx = sl.backward()    
    print("dx\n", dx)       
    