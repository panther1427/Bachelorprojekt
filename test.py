import numpy as np
a = np.array([1,2,3])
b = a.copy()
b[1] = 999

print(a)
print(b)