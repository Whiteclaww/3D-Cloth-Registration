#========[ IMPORTS ]========
from modules.utils.conversion import toarray_without_loss

import logging
import scipy.sparse as sp
from scipy.sparse.linalg import lsmr

#========[ FUNCTIONS ]========

def spsolve(A, B):
    r"""
    Solve A*X = B (matrices)
    Since B is much smaller than A, we divide B in multiple vectors, calculate A*X = b then append every X to the result

    Args:
        A (sparse matrix)
        B (sparse matrix)
    """
    X = []
    tol = 0#1e-17
    for count in range(B.shape[1]):
        # Extract the i-th row of b
        b = B[:, count].reshape(1, -1)
        
        # Solve the linear system for the i-th right-hand side vector
        (solved_row_b, stop, _, _, _, _, _, _) = lsmr(A, toarray_without_loss(b), atol=tol, btol=tol, damp=0, conlim=0)
        
        if stop != 5 and stop != 2:
            logging.error("Error Solving Ax=B")
        
        # Append the solution to the list of solutions
        X.append(solved_row_b)
        
    X = sp.csr_matrix(X).tocsr().T
    
    '''maybeB:sp.csr_matrix = A.dot(X)
    
    abs_diff = np.abs(toarray_without_loss(maybeB) - toarray_without_loss(B))
    if np.all(abs_diff <= tolerance).all() and maybeB.shape == B.shape:
        logging.info(" |  | Verified that A * X = B")
    else:
        logging.error("=====Error when verifying A * X = B, precision error")'''
        
    return X