
import os
import math
import time
import struct
import json
import numpy as np
import numpy.linalg as npla
import scipy
import scipy.sparse.linalg as spla
from scipy import sparse
from scipy import linalg


def pagerank2(E, return_vector = False, max_iters = 1000, tolerance = 1e-6):
    """compute page rank from dense adjacency matrix

    Inputs:
      E: adjacency matrix with links going from cols to rows.
         E is a matrix of 0s and 1s, where E[i,j] = 1 means 
         that web page (vertex) j has a link to web page i.
      return_vector = False: If True, return the eigenvector as well as the ranking.
      max_iters = 1000: Maximum number of power iterations to do.
      tolerance = 1e-6: Stop when the eigenvector norm changes by less than this.
      
    Outputs:
      ranking: Permutation giving the ranking, most important first
      vector (only if return_vector is True): Dominant eigenvector of PageRank matrix

    This computes page rank by the following steps:
    1. Add links from any dangling vertices to all vertices.
    2. Scale the columns to sum to 1.
    3. Add a constant matrix to represent jumping at random 15% of the time.
    4. Find the dominant eigenvector with the power method.
    5. Sort the eigenvector to get the rankings.

    The homework problem due February 22 asks you to rewrite this code so
    it takes input E as a scipy csr_sparse matrix, and then never creates 
    a full matrix or any large matrix other than E.
    """
    """
    if type(E) is not np.ndarray:
        print('Warning, converting input from type', type(E), 'to dense array.')
        E = E.toarray()
    """
    np.seterr(divide='ignore', invalid = 'ignore')
    nnz = E.nnz # This call for sparse E may be different
    outdegree = np.sum(E, 0).A1  # This call for sparse E may be different
    #D = np.diag(1/outdegree)
    nrows, n = E.shape

    assert nrows == n, 'E must be square'
    assert np.max(E) == 1 and np.sum(E) == nnz, 'E must contain only zeros and ones'
    
    #preparation to save memory
    #E = E@D now is link matrix
    #print(E.toarray())
    ZeroCol = np.where(outdegree == 0)
    e = np.ones(n)
    v = e / npla.norm(e)
    for iteration in range(max_iters):
        oldv = v
        #E*V   E = E/np.sum(E,0)
        v = E.dot(0.85*oldv/outdegree)
        v += 0.15/n*sum(oldv)
        #F*v
        v += sum(oldv[ZeroCol]*(0.85/(n-1)))
        v[ZeroCol] -= oldv[ZeroCol]*(0.85/(n-1))
        v = v/npla.norm(v)
        if npla.norm(v - oldv) < tolerance:
            eigval = np.average(oldv/v)
            break
    
    if npla.norm(v - oldv) < tolerance:
        print('Dominant eigenvalue is %f after %d iterations.\n' % (eigval, iteration+1))
    else:
        print('Did not converge to tolerance %e after %d iterations.\n' % (tolerance, max_iters))
    # Check that the eigenvector elements are all the same sign, and make them positive
    assert np.all(v > 0) or np.all(v < 0), 'Error: eigenvector is not all > 0 or < 0'
    vector = np.abs(v)
        
    #  5. Sort the eigenvector and reverse the permutation to get the rankings.
    ranking = np.argsort(vector)[::-1]
    if return_vector:
        return ranking, vector
    else:
        return ranking
