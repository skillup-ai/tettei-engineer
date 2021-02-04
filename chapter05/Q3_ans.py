import numpy as np

class Standardization:
    def fit_transform(self, X):
        """
        X.shape = (データ数, 次元数)
        """

        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        Xsd = (X - self.mean_) / self.std_
        return Xsd

    def transform(self, X):
        """
        X.shape = (データ数, 次元数)
        """

        Xsd = (X - self.mean_) / self.std_
        return Xsd

    def inverse_transform(self, Xsd):
        """
        Xsd.shape = (データ数, 次元数)
        """

        X = (Xsd * self.std_) + self.mean_
        return X


if __name__=="__main__":
    
    std = Standardization()
    
    X = np.array([[1,2,3],
                     [4,5,6]])
    print("X\n", X)
    
    Xsd = std.fit_transform(X)
    print("Xsd\n", Xsd)
    
    Xsd = std.transform(X)
    print("Xsd\n", Xsd)
    
    X_ = std.inverse_transform(Xsd)
    print("X_\n", X_)
    