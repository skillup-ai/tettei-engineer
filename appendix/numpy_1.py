import numpy as np

a = np.array([[1, 3], [5, 7]])
b = np.array([[1, 2], [3, 4]])

print(a * b) #乗算（アダマール積）：np.multiply(a, b) でも同様の結果が得られる

'''
[[ 1  6]
 [15 28]]
'''

print(np.square(a)) #二乗：a ** 2 でも同様の結果が得られる

'''
[[ 1  9]
 [25 49]]
'''

print(np.power(a, 3)) #べき乗：a ** 3 でも同様の結果が得られる

'''
[[  1  27]
 [125 343]]
'''