def cross_entropy_error(y, t):
    """
    y : ソフトマックス関数の出力
        y.shape=(k,)またはy.shape=(N,k)
    t : 正解ラベル(ワンホット表現)
        t.shape=(k,)またはt.shape=(N,k)
    """
    if y.ndim==1:
        t = t.reshape(1,-1)
        y = y.reshape(1,-1)

    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(t*np.log(y + delta))/ batch_size
