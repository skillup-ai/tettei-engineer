import numpy as np

x_1 = np.array([1, 2, 3])
x_2 = np.array([[1, 2, 3]])


print(x_1.ndim) #一次元 array

'''
1
'''

print(x_2.ndim) #二次元 array

'''
2
'''

print(x_1.T)

'''
[1 2 3]
'''

print(x_2.T)

'''
[[1]
 [2]
 [3]]
'''