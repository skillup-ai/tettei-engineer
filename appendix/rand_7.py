import numpy as np

np.random.seed(1000) # シードの固定

s = ['a', 'b', 'c', 'd', 'e']

a = np.random.permutation(s)

print(s) # 元のリストは置き換わっていない

'''
['a', 'b', 'c', 'd', 'e']
'''

print(a)

'''
['c', 'b', 'a', 'e', 'd']
'''