import numpy as np


def im2col(input_data, filter_h, filter_w, stride, pad, constant_values=0):
    """
    input_data : (データ数, チャンネル数, 高さ, 幅)の4次元配列
    filter_h : フィルタの高さ
    filter_w : フィルタの幅
    stride : ストライドサイズ
    pad : パディングサイズ
    constant_values : パディング処理で埋める際の値
    """

    # 入力データのデータ数, チャンネル数, 高さ, 幅を取得
    N, C, H, W = input_data.shape

    # 出力データの高さ(端数は切り捨てる)
    out_h = (H + 2 * pad - filter_h) // stride + 1

    # 出力データの幅(端数は切り捨てる)
    out_w = (W + 2 * pad - filter_w) // stride + 1

    # パディング処理
    img = np.pad(
        input_data,
        [(0, 0), (0, 0), (pad, pad), (pad, pad)],
        "constant",
        constant_values=constant_values,
    )

    # 配列の初期化
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    # フィルタ内のある1要素に対応する画像中の画素を取り出してcolに代入
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    # 軸を入れ替えて、2次元配列(行列)に変換する
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)

    return col


def maxpooling_forward(x, pad, stride, pool_h, pool_w):
    """
    x : 入力データ, 配列形状 = (データ数, チャンネル数, 高さ, 幅)
    pad : パディングサイズ
    stride : ストライドサイズ
    pool_h : プーリング領域の縦
    pool_w : プーリング領域の横
    """

    N, C, H, W = x.shape

    # 出力の高さ(端数は切り捨てる)
    out_h = (H + 2 * pad - pool_h) // stride + 1

    # 出力の幅(端数は切り捨てる)
    out_w = (W + 2 * pad - pool_w) // stride + 1

    # 2次元配列に変換する
    col = im2col(x, pool_h, pool_w, stride, pad, constant_values=0)

    # チャンネル方向のデータが横に並んでいるので、縦に並べ替える
    col = col.reshape(-1, pool_h * pool_w)

    # 最大値と最大値のインデックス（逆伝播時に使用）を求める
    out_idx = np.argmax(col, axis=1)

    # 最大値を求める
    out = np.max(col, axis=1)

    # 画像形式に戻して、チャンネルの軸を2番目に移動させる
    out = out.reshape(N, out_h, out_w, C). transpose(0, 3, 1, 2)

    return out_idx, out


def convolution_forward(x, W, b, pad, stride):
    """
    x : 入力データ, 配列形状 = (データ数, チャンネル数, 高さ, 幅)
    W : フィルタ, 配列形状 = (出力チャンネル数, 入力チャンネル数, 高さ, 幅)
    b : バイアス
    pad : パディングサイズ
    stride : ストライドサイズ
    """

    FN, C, FH, FW = W.shape
    N, C, IH, IW = x.shape

    # 出力の高さ(端数は切り捨てる)
    out_h = (IH + 2 * pad - FH) // stride + 1

    # 出力の幅(端数は切り捨てる)
    out_w = (IW + 2 * pad - FW) // stride + 1

    # 畳み込み演算を効率的に行えるようにするため、入力xを行列colに変換する
    col = im2col(x, FH, FW, stride, pad)

    # フィルタを2次元配列に変換する
    col_W = W.reshape(FN, -1).T

    # 行列の積を計算し、バイアスを足す
    out = np.dot(col, col_W) + b

    # 画像形式に戻して、チャンネルの軸を2番目に移動させる
    out = out.reshape(N, out_h, out_w, -1). transpose(0, 3, 1, 2)

    return out


if __name__=="__main__":
    x = np.arange(2*3*28*28).reshape(2, 3, 28 ,28)
    print("x\n", x)
    pad = 1
    stride = 1
    pool_h = 3
    pool_w = 3
    out_idx, out = maxpooling_forward(x, pad, stride, pool_h, pool_w)
    print("out_idx\n", out_idx)
    print("out\n", out)
    
    x = np.arange(2*3*28*28).reshape(2, 3, 28 ,28)
    print("x\n", x)    
    W = np.arange(5*3*3*3).reshape(5, 3, 3, 3)
    print("W\n", W) 
    b = np.array([1])
    print("b\n", b)
    pad = 1
    stride = 1    
    out = convolution_forward(x, W, b, pad, stride)