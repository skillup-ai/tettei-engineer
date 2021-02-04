import numpy as np

np.random.seed(1000) # シードの固定

s = ['a', 'b', 'c', 'd', 'e']

np.random.shuffle(s) #シャッフルしたリストで置き換える

print(s)

'''
['c', 'b', 'a', 'e', 'd']
'''