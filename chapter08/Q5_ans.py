import numpy as np

def pca(X, n_components=2):

    # データから平均を引く
    X = X - X.mean(axis=0)

    # 共分散行列の作成
    cov = np.cov(X, rowvar=False)

    # 固有値や主成分方向を計算
    l, v = np.linalg.eig(cov)

    # 固有値の大きい順に並び替え
    l_index = np.argsort(l)[::-1]
    v_ = v[:,l_index]

    # n_components分，主成分方向を取得
    components = v_[:,:n_components]

    # データを低次元空間へ射影
    T = np.dot(X, components)

    return T

