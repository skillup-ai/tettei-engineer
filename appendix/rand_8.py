import numpy as np

np.random.seed(1000) # シードの固定

a = np.random.permutation(10) # 0 以上 10 未満の整数列をシャッフル

print(a)

'''
[2 6 5 1 4 9 0 8 7 3]
'''