from scipy import sparse

def make_A_3D(k):
    """
    Create the matrix for the temperature problem on a k-by-k-by-k 3D grid.
    Parameters:
        k: number of grid points in each dimension.
    Outputs:
        A: the sparse k**3-by-k**3 matrix representing the finite difference approximation to Poisson's equation.
    """
    triples = []
    for l in range(k): #level
        for r in range(k): #row
            for c in range(k): #col
                row = c + r*k + l*k*k
                triples.append((row,row, 6.0))
                #connect to left
                if c>0:
                    triples.append((row,row-1,-1.0))
                #connect to right
                if c<k-1:
                    triples.append((row,row+1,-1.0))
                #connect to last row
                if r>0:
                    triples.append((row,row-k,-1.0))
                #connect to next row
                if r<k-1:
                    triples.append((row,row+k,-1.0))
                #connect to last level
                if l>0:
                    triples.append((row,row-k*k,-1.0))
                if l<k-1:
                    triples.append((row,row+k*k,-1.0))
    #convert list of triples to a scipy sparse matrix
    ndim = k*k*k
    rownum = [t[0] for t in triples]
    colnum = [t[1] for t in triples]
    values = [t[2] for t in triples]
    A = sparse.csr_matrix((values,(rownum, colnum)), shape = (ndim, ndim))
    
    return A

