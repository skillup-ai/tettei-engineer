import numpy as np
from Q2_ans import compute_distances


def init_centroid(X, n_data, k):
    # 1つ目の重心をランダムに選択
    idx = np.random.choice(n_data, 1)
    centroids = X[idx]
    for i in range(k - 1):
        # 各データ点と重心との距離を計算
        distances = compute_distances(X, len(centroids), n_data, centroids)

        # 各データ点と最も近い重心との距離の二乗を計算
        closest_dist_sq = np.min(distances ** 2, axis=1)

        # 距離の二乗の和を計算
        weights = closest_dist_sq.sum()

        # [0,1)の乱数と距離の二乗和を掛ける
        rand_vals = np.random.random_sample() * weights

        # 距離の二乗の累積和を計算し，rand_valと最も値が近いデータ点のindexを取得
        candidate_ids = np.searchsorted(np.cumsum(closest_dist_sq), rand_vals)

        # 選ばれた点を新たな重心として追加
        centroids = np.vstack([centroids, X[candidate_ids]])
    return centroids

if __name__=="__main__":
    X = np.arange(10*5).reshape(10, 5)
    print("X\n", X)
    
    k = 3 
    print("k=",k)
    
    n_data = X.shape[0]
    centroids = init_centroid(X, n_data, k)        
    print("centroids", centroids)