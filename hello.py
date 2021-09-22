msg = "Hello World"
print(msg)

import numpy as np

N = np.array([1,1,1])
n = N / np.linalg.norm(N)
print(n)

reflector = np.add(np.identity(3),-2*np.outer(n,n))
print(reflector)
eigen = np.linalg.eigh(reflector)
print(eigen[0])
print(np.transpose(eigen[1]))
print(np.transpose(eigen[1])[0])