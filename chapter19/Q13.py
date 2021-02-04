class Adam:
    """
    Adam
    """
    def __init__(self, lr=0.001, rho1=0.9, rho2=0.999):
        self.lr = lr
        self.rho1 = rho1
        self.rho2 = rho2
        self.iter = 0
        self.m = None # 1次モーメント. 勾配の平均に相当する
        self.v = None  # 2次モーメント. 勾配の分散に相当する(中心化されていない)
        self.epsilon = 1e-8
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        
        for key in params.keys():
            self.m[key] = self.rho1 * self.m[key] + (1 - self.rho1) * grads[key] # 1次モーメント
            self.v[key] = self.rho2 * self.v[key] + (1 - self.rho2) * (grads[key] ** 2) #2次モーメント
            
            # モーメントのバイアス補正
            # 計算初期の頃のモーメントを補正することが目的
            # 計算が進行する(self.iterが大きくなる)と、分母は1に近づく
            m = self.m[key] / (1 - self.rho1 ** self.iter) # 1次モーメント
            v = self.v[key] / (1 - self.rho2 ** self.iter) # 2次モーメント
            
            # 重みの更新
            params[key] -= self.lr * m / (np.sqrt(v) + self.epsilon) 