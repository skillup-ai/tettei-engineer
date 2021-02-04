import numpy as np

class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask

    
if __name__=="__main__":
    x = np.arange(2*3).reshape(2, 3)
    print("x\n", x)

    do = Dropout()
    out = do.forward(x)
    print("out\n", out)
    
    out = do.forward(x, train_flg=False)
    print("out\n", out)
    
    dout = np.arange(2*3).reshape(2, 3)
    print("dout\n", dout)        
    dx = do.backward(dout)
    print("dx", dx)    