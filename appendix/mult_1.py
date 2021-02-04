import numpy as np

x = np.array([[1, 3, 5], [7, 9, 11]])
y = np.array([[1, 2], [3, 4], [5, 6]])

print(np.dot(x, y)) #行列積

'''
[[ 35  44]
 [ 89 116]]
'''

print(x.dot(y)) #行列積

'''
[[ 35  44]
 [ 89 116]]
'''

print(x @ y) #行列積

'''
[[ 35  44]
 [ 89 116]]
'''