import numpy as np

np.random.seed(1000) # シードの固定

s = ['a', 'b', 'c', 'd', 'e']

a = np.random.choice(s, 10)

print(a)

'''
['d' 'a' 'd' 'e' 'b' 'a' 'b' 'a' 'b' 'e']
'''