import numpy as np

np.random.seed(1000) # シードの固定

print(np.random.binomial(100, 0.3)) # n=100, p=0.3 の二項乱数

'''
32
'''

print(np.random.binomial(100, 0.3, 10)) # n=100, p=0.3 の二項乱数を 10 個

'''
[25 38 30 35 26 22 29 27 35 26]
'''