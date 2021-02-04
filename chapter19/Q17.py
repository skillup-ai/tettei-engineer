import numpy as np


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters
    ----------
    input_data : (データ数, チャンネル, 高さ, 幅)の4次元配列からなる入力データ
    filter_h : 畳み込みカーネルの高さ
    filter_w : 畳み込みカーネルの幅
    stride : 畳み込みのストライド幅
    pad : 畳み込みのパディングサイズ 
    Returns
    -------
    col : 2次元配列
    """
    
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], "constant")
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(out_h * out_w * N, -1)
    return col


if __name__=="__main__":
    N = 2
    C = 3
    H = 4
    W = 4
    input_data = np.random.randn(N, C, H, W)
    print("input_data\n", input_data)
    
    filter_h = 3
    filter_w = 3
    col = im2col(input_data, filter_h, filter_w)
    print("col\n", col)

    