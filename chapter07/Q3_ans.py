import numpy as np

class Sigmoid:
    def forward(self, x, w, b):
        """
        x.shape = (データ数, 次元数)
        w.shape = (1, 次元数)
        b.shape = (1,)
        """
        self.x = x
        z = np.sum(w * x,  axis=1) + b
        y_pred = 1 / (1 + np.exp(-z))
        self.y_pred = y_pred
        return y_pred

    def backward(self, dy):
        """
        dy.shape = (データ数, )
        """
        dz = dy * (1.0 - self.y_pred) * self.y_pred
        dw = np.sum(self.x * dz.reshape(-1,1), axis=0)
        db = np.sum(dz)
        return dw, db

    
class NegativeLogLikelihood:
    def forward(self, y_pred, y_true):
        """
        y_pred.shape = (データ数,)
        y_true.shape = (データ数,)
        """
        self.y_pred = y_pred
        self.y_true = y_true
        loss = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss.sum()

    def backward(self):
        dy = - (self.y_true / self.y_pred) + ((1 - self.y_true) / (1 - self.y_pred))
        return dy

    
if __name__=="__main__":
    
    sg = Sigmoid()
    
    x = np.random.randn(10, 3)
    w = np.random.randn(1, 3)
    b = np.array([1.0])
    print("x\n", x)
    print("w\n", w)
    print("b\n", b)
    
    y_pred = sg.forward(x, w, b)
    print("y_pred\n", y_pred)

    nll = NegativeLogLikelihood()
    y_true = x[:, 0] > 0
    print("y_true\n", y_true)
    
    loss = nll.forward(y_pred, y_true)
    print("loss",loss)
    
    dy = nll.backward()
    print("dy", dy)
    
    dw, db = sg.backward(dy)
    print("dw\n",dw)
    print("db\n",db)    

    # パラメータw,bを勾配法によって最適化する
    lr = 0.1
    for i in range(1000):
        y_pred = sg.forward(x, w, b)
        loss = nll.forward(y_pred, y_true)
        print("loss=", loss)
        dy = nll.backward()
        dw, db = sg.backward(dy)
        w -= lr * dw
        b -= lr * db
    print(w,b)
    print("w[0]の値だけ大きくなっていたら学習成功\n")
    
    print("訓練データに対する予測結果")
    y_pred = sg.forward(x, w, b) 
    y_pred = y_pred > 0.5
    print("y_pred\n", y_pred)
    print("y_true\n", y_true)