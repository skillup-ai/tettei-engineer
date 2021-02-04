import numpy as np

x = np.array([[1, 3, 5], [2, 4, 6]]) #二次元 array とスカラーの和
y = 2

print(x + y)

'''
[[3 5 7]
 [4 6 8]]
'''

x = np.array([[1, 3, 5], [2, 4, 6]]) #二次元 array と一次元 array の和
y = np.array([10, 20, 30])

print(x + y)

'''
[[11 23 35]
 [12 24 36]]
'''