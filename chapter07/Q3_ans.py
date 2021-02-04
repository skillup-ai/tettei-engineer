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
    
    x = np.array([[1,2,3],
                    [4,5,6]])
    w = np.array([[1,2,3]])
    b = np.array([1])
 
    y_pred = sg.forward( x, w, b)
    print("y_pred\n", y_pred)
    
    dy = np.array([1,2])
    dw, db = sg.backward(dy)
    print("dw\n",dw)
    print("db\n",db)
    
    nll = NegativeLogLikelihood()
    y_true = np.array([1,2])
    
    loss = nll.forward(y_pred, y_true)
    print("loss",loss)
    
    dy = nll.backward()
    print("dy", dy)
    