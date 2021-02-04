import numpy as np

a = np.array([1,2,3,4,5])

print(np.searchsorted(a, 3)) # a[2] の左に挿入すると順序を破壊しない

'''
2
'''

print(np.searchsorted(a, [-10, 10, 2, 3])) # 4 つの要素それぞれに対して挿入すべきインデックスを返す

'''
[0, 5, 1, 2]
'''