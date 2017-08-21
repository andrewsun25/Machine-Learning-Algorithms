from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import my_utilities as my


def reduce_dimensions(X, min_var=0.9):
    X = (X - X.mean()) / X.std()
    cov = 1. / X.shape[0] * X.T.dot(X)
    U, S, V = np.linalg.svd(cov)
    n = len(S)
    k = 1
    for k in range(1, n):
        if np.sum(S[:k]) / np.sum(S) >= min_var:
            break
    U_reduce = U[:, :k]
    z = X.dot(U_reduce)
    return U_reduce, z

def recover_dimensions(U_reduce, Z):
    return Z.dot(U_reduce.T)

# X shape(5000, 1024) (mxn)
X = loadmat('data/ex7faces.mat')['X']
# face shape (32, 32)
face = np.reshape(X[420,:], (32, 32))
# U_reduce shape (32,4) and z shape (32,4)
U_reduce, face_reduced = reduce_dimensions(face)
face_recovered = recover_dimensions(U_reduce, face_reduced)
my.cool_print(U_reduce=U_reduce, face_reduced=face_reduced,face=face, face_recovered=face_recovered, pprint=False)
plt.imshow(face, cmap='gray')
plt.show()
plt.imshow(face_recovered, cmap='gray')
plt.show()
