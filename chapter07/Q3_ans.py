import numpy as np

class Sigmoid:
    def forward(self, x, w, b):
        """
        x.shape = (データ数, 次元数)
        w.shape = (1, 次元数)
        b.shape = (1,)
        """

        self.x = x
        z = np.sum(w * x, axis=1) + b
        y_pred = 1 / (1 + np.exp(-z))
        self.y_pred = y_pred
        return y_pred

    def backward(self, dy):
        dz = dy * (1.0 - self.y_pred) * self.y_pred
        dw = np.sum(dz.T * self.x, axis=0)
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
