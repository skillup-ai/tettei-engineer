import numpy as np

np.random.seed(1000) # シードの固定

print(np.random.randint(100, 1000)) # 100 以上 1000 未満の整数一様乱数

'''
535
'''

print(np.random.randint(100, 1000, 10)) # 100 以上 1000 未満の整数一様乱数を 10 個

'''
[699 171 804 351 450 448 869 545 740 957]
'''