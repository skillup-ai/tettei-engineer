epoch = 100  # エポック数
batch_size = 32
learning_rate = 0.01

# 訓練データの数を取得
train_size = X_train.shape[0]

# 1エポックにおけるイテレーションの回数を計算
iter_per_epoch = max(train_size//batch_size, 1)

# パラメータの初期化
weights = np.random.randn(X_train.shape[1])

# 学習
for i in range(epoch):
    # 訓練データを全件、入出力の対応関係を保ったままシャッフルする
    p = np.random.permutation(train_size)
    X_train = X_train[p]
    y_train = y_train[p]

    for j in range(iter_per_epoch):
        # シャッフルしてきた訓練データの中から、先頭から順にミニバッチを取得してくる
        batch_start_index = j * batch_size
        batch_end_index = batch_start_index + batch_size
        if batch_end_index > train_size:
            batch_end_index = train_size
        X_batch = X_train[batch_start_index: batch_end_index]
        y_batch = y_train[batch_start_index: batch_end_index]

        # 勾配の計算
        grads = gradients(weights, X_batch, y_batch)
        # 重みの更新
        weights = weights - learning_rate * grads
