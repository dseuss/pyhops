#ifndef SPARSETOOLS_HPP_FI2SPZYG
#define SPARSETOOLS_HPP_FI2SPZYG

#include <algorithm>
/*
* Compute B = A for COO matrix A, CSR matrix B
*
*
* Input Arguments:
* I n_row - number of rows in A
* I n_col - number of columns in A
* I nnz - number of nonzeros in A
* I Ai[nnz(A)] - row indices
* I Aj[nnz(A)] - column indices
* T Ax[nnz(A)] - nonzeros
* Output Arguments:
* I Bp - row pointer
* I Bj - column indices
* T Bx - nonzeros
*
* Note:
* Output arrays Bp, Bj, and Bx must be preallocated
*
* Note:
* Input: row and column indices *are not* assumed to be ordered
*
* Note: duplicate entries are carried over to the CSR represention
*
* Complexity: Linear. Specifically O(nnz(A) + max(n_row,n_col))
*
* Taken from scipy/sparse/coo.h
*/
template <class I, class T>
void coo_tocsr(const I n_row,
               const I n_col,
               const I nnz,
               const I Ai[],
               const I Aj[],
               const T Ax[],
                     I Bp[],
                     I Bj[],
                     T Bx[])
{
    //compute number of non-zero entries per row of A
    std::fill(Bp, Bp + n_row, 0);

    for (I n = 0; n < nnz; n++){
        Bp[Ai[n]]++;
    }

    //cumsum the nnz per row to get Bp[]
    for(I i = 0, cumsum = 0; i < n_row; i++){
        I temp = Bp[i];
        Bp[i] = cumsum;
        cumsum += temp;
    }
    Bp[n_row] = nnz;

    //write Aj,Ax into Bj,Bx
    for(I n = 0; n < nnz; n++){
        I row = Ai[n];
        I dest = Bp[row];

        Bj[dest] = Aj[n];
        Bx[dest] = Ax[n];

        Bp[row]++;
    }

    for(I i = 0, last = 0; i <= n_row; i++){
        I temp = Bp[i];
        Bp[i] = last;
        last = temp;
    }

    //now Bp,Bj,Bx form a CSR representation (with possible duplicates)
}

#endif /* end of include guard: SPARSETOOLS_HPP_FI2SPZYG */
