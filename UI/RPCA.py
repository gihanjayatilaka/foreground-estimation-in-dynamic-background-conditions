import numpy as np
from numpy.linalg import svd
from numpy.linalg import norm
from scipy.linalg import qr
import cv2

def inexact_augmented_lagrange_multiplier(X, lmbda=.01, tol=1e-3, maxiter=100, verbose=True):
    """
    Inexact Augmented Lagrange Multiplier
    """
    Y = X
    norm_two = norm(Y.ravel(), 2)
    norm_inf = norm(Y.ravel(), np.inf) / lmbda
    dual_norm = np.max([norm_two, norm_inf])
    Y = Y / dual_norm
    A = np.zeros(Y.shape)
    E = np.zeros(Y.shape)
    dnorm = norm(X, 'fro')
    mu = 1.25 / norm_two
    rho = 1.5
    sv = 10.
    n = Y.shape[0]
    itr = 0
    while True:
        Eraw = X - A + (1 / mu) * Y
        Eupdate = np.maximum(Eraw - lmbda / mu, 0) + np.minimum(Eraw + lmbda / mu, 0)
        U, S, V = svd(X - Eupdate + (1 / mu) * Y, full_matrices=False)
        svp = (S > 1 / mu).shape[0]
        if svp < sv:
            sv = np.min([svp + 1, n])
        else:
            sv = np.min([svp + round(.05 * n), n])
        Aupdate = np.dot(np.dot(U[:, :svp], np.diag(S[:svp] - 1 / mu)), V[:svp, :])
        A = Aupdate
        E = Eupdate
        Z = X - A - E
        Y = Y + mu * Z
        mu = np.min([mu * rho, mu * 1e7])
        itr += 1
        if ((norm(Z, 'fro') / dnorm) < tol) or (itr >= maxiter):
            break
    if verbose:
        print("Finished at iteration %d" % (itr))
    return A, E

'''
cap = cv2.VideoCapture('sample3.mp4')

n_frame = 200
mat = []
m,n,o = 0,0,0


if (cap.isOpened()== False):
  print("Error opening video stream or file")
else:
    ret, frame = cap.read()
    m, n, o = frame.shape
print(m,n)


for i in range(n_frame):
#while(cap.isOpened()):
    ret, frame = cap.read()

    if ret == True:
        f = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Frame',f)

        mat.append(np.ndarray.flatten(f))
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()

cv2.destroyAllWindows()

matrix = np.asarray(mat).T
print (matrix.shape)

print(matrix)
print(matrix.dtype)

print(np.amax(matrix, axis=None))


A, E = inexact_augmented_lagrange_multiplier(matrix, lmbda=.002)
E = E.astype(np.uint8)
A = A.astype(np.uint8)

print(E)
print(E.dtype)

print(np.amax(E, axis=None))

play(A.T, m, n)
play(E.T, m, n)
'''
'''
L, S, G = go_dec(matrix, rank=5, max_iter=200)

L = L.astype(np.uint8)
S = S.astype(np.uint8)
G = G.astype(np.uint8)

play(L.T, m, n)
play(S.T, m, n)
play(G.T, m, n)
'''
