import numpy as np
from common.functions import softmax

class Softmax:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        self.out = softmax(x)
        return self.out

    def backward(self, dout):
        dx = self.out*dout
        sumdx = np.sum(dx,axis=1,keepdims=True)
        dx -= self.out*sumdx
        return dx

