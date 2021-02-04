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

    
if __name__=="__main__":
    x = np.arange(2*3).reshape(2, 3)
    print("x\n", x)
    
    sf = Softmax()   
    out = sf.forward(x)    
    print("out\n", out)
    
    dout = np.arange(2*3).reshape(2, 3)
    print("dout\n", dout)    
    dx = sf.backward(dout)    
    print("dx\n", dx)    