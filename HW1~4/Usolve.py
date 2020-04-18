import numpy as np
 
def Usolve(U, y):
    """Backward solve an upper triangular system Ux = y for x
    Parameters:
      U: the matrix, must be square, upper triangular, with nonzeros on the diagonal
      y: the right-hand side vector
    Output:
      x: the solution vector to U @ x == y
    """
    # Check the input
    m, n = U.shape
    assert m == n, "matrix L must be square"
    assert np.all(np.triu(U) == U), "matrix L must be lower triangular"
    assert np.all(np.diag(U)!=0), "matrix must has nonezero diagonal"
    x = y.astype(np.float64).copy()
    U = 1.0*U
    for col in reversed(range(n)):
        x[col]=x[col]/U[col, col]
        U[col,:]=U[col,:]/U[col,col]
    for col in reversed(range(n)):
        x[:col]= x[:col]-x[col]*U[:col,col]
    return x
