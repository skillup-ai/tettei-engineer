import numpy as np

def init_centroid(X, n_data, k):
    # 各データ点の中からクラスタの重心となる点をk個ランダムに選択
    idx = np.random.permutation(n_data)[:k]
    centroids = X[idx]
    return centroids


def compute_distances(X, k, n_data, centroids):
    distances = np.zeros((n_data, k))
    for idx_centroids in range(k):
        dist = np.sqrt(np.sum((X - centroids[idx_centroids]) ** 2, axis=1))
        distances[:, idx_centroids] = dist
    return distances


def k_means(k, X, max_iter=300):
    """
    X.shape = (データ数, 次元数)
    k = クラスタ数
    """
    n_data, n_features = X.shape

    # 重心の初期値
    centroids = init_centroid(X, n_data, k)

    # 新しいクラスタを格納するための配列
    new_cluster = np.zeros(n_data)

    # 各データの所属クラスタを保存する配列
    cluster = np.zeros(n_data)

    for epoch in range(max_iter):
        # 各データ点と重心との距離を計算
        distances = compute_distances(X, k, n_data, centroids)

        # 新たな所属クラスタを計算
        new_cluster = np.argmin(distances, axis=1)

        # すべてのクラスタに対して重心を再計算
        for idx_centroids in range(k):
            centroids[idx_centroids] = X[new_cluster == idx_centroids].mean(axis=0)

        # クラスタによるグループ分けに変化がなかったら終了
        if (new_cluster == cluster).all():
            break

        cluster = new_cluster

    return cluster


if __name__=="__main__":
    X = np.arange(10*5).reshape(10, 5)
    print("X\n", X)
    k = 3 
    print("k=",k)

    cluster = k_means(k, X, max_iter=300)
    print("cluster", cluster)
    
    
    
    
    