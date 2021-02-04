list_1 = ["a", "b", "c", "d"]
list_2 = ["x", "y", "z", "w"]

for foo, bar in zip(list_1, list_2):
    print(f"{foo}, {bar}")

'''
a, x
b, y
c, z
d, w
'''