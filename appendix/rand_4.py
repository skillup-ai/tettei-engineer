import numpy as np

np.random.seed(1000) # シードの固定

print(np.random.randn()) # 平均が 0 標準偏差が 1 の正規乱数

'''
-0.8044583035248052
'''

print(np.random.randn(10)) # 平均が 0 標準偏差が 1 の正規乱数を 10 個

'''
[ 0.32093155 -0.02548288  0.64432383 -0.30079667  0.38947455
 -0.1074373 -0.47998308  0.5950355  -0.46466753  0.66728131]
'''