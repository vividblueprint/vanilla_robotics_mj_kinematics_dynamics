import numpy as np

A = np.array(
    [
        [2,4],
        [5,-6]
    ]
)

b = np.array(
    [7,10]
)

c = A.dot(b)
Ainv = np.linalg.inv(A)
I = np.identity(4)

print(A)
print(b)
print(c)

print(type(A))
print(np.shape(A))
print(np.shape(b))

print(A.transpose())
print(A.T)
print(Ainv)
print(I)
print(np.matmul(A,Ainv))