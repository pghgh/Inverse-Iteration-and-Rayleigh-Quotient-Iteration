import time
import numpy as np
import scipy as sp
from scipy import linalg


def inverse_iteration_with_shift_function(A, sigma, seed, mach_eps, max_no_of_iterations, x = None):

    start = time.time()
    print("Start of method: inverse_iteration_with_shift_function")

    rng = np.random.default_rng(seed)
    n = len(A)
    I = np.eye(n)
    M = A - sigma * I
    if not sp.linalg.det(M):
        sigma += 0.1
        M = A - sigma * I

    P, L, U = sp.linalg.lu(M)
    P = P.transpose()

    if x is None:
        x = rng.random(n)[np.newaxis]
    x = x.transpose()

    res = 1
    eigenval = 0
    iterations = 0

    while res > mach_eps and iterations <= max_no_of_iterations:
        iterations += 1
        eigenval = (x.transpose() @ A @ x) / (x.transpose() @ x)
        z = sp.linalg.solve_triangular(L, np.dot(P, x), lower=True, unit_diagonal=True)
        y = sp.linalg.solve(U, z)
        x = y / sp.linalg.norm(y, np.inf)
        res = sp.linalg.norm(A @ x - eigenval * x, 2)

    end = time.time()
    elapsed_time = end - start

    return eigenval, x, iterations, res, elapsed_time
