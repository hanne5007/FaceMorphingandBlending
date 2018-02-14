'''
  File name: getCoefficientMatrix.py
  Author:
  Date created:
'''

from scipy.sparse import csr_matrix
import numpy as np

def getCoefficientMatrix(indexes):
    [h,w]=np.shape(indexes)
    #indexes=indexes.astype(np.int64)
    # number of replacement pixels
    N = np.count_nonzero(indexes)
    #coeffA=np.zeros([N,N])

    # sparse csr matrix
    coeffA=csr_matrix((N,N))

    # entries on the diagonal equal to 4
    #row_dia=np.arange(N)
    #col_dia=np.arange(N)
    #dia=4*np.ones([1,N])
    #dia=dia[0]
    #coeffA[dia, dia]=4

    ind=indexes.nonzero()
    row=ind[0]
    col=ind[1]
    #next_col=col+1
    row_n=[]
    col_n=[]
    val=[]

    # recursion on every replacement pixel
    for i in range(0, N):
        row_n.append(indexes[row[i], col[i]] - 1)
        col_n.append(indexes[row[i], col[i]] - 1)
        val.append(4)
        if col[i] + 1 < w and indexes[row[i], col[i] + 1] != 0:
            row_n.append(indexes[row[i], col[i]] - 1)
            col_n.append(indexes[row[i], col[i] + 1] - 1)
            val.append(-1)
        if row[i] + 1 < h and indexes[row[i] + 1, col[i]] != 0:
            row_n.append(indexes[row[i], col[i]] - 1)
            col_n.append(indexes[row[i] + 1, col[i]] - 1)
            val.append(-1)
        if col[i] - 1 >= 0 and indexes[row[i], col[i] - 1] != 0:
            row_n.append(indexes[row[i], col[i]] - 1)
            col_n.append(indexes[row[i], col[i] - 1] - 1)
            val.append(-1)
        if row[i] - 1 >= 0 and indexes[row[i] - 1, col[i]] != 0:
            row_n.append(indexes[row[i], col[i]] - 1)
            col_n.append(indexes[row[i] - 1, col[i]] - 1)
            val.append(-1)

    row_n=np.array(row_n)
    col_n=np.array(col_n)
    val=np.array(val)

    coeffA = csr_matrix((val, (row_n, col_n)), shape = (N, N))
    #c=coeffA.toarray()
    return coeffA
