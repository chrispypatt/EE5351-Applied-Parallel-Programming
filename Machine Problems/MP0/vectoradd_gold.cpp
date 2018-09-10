#include <stdlib.h>
////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C"
void computeGold( float*, const float*, const float*, unsigned int);

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! C = A * B
//! @param C          reference data, computed but preallocated
//! @param A          vector A as provided to device
//! @param B          vector B as provided to device
//! @param N         length of vectors
////////////////////////////////////////////////////////////////////////////////
void
computeGold(float* C, const float* A, const float* B, unsigned int N)
{
    for (unsigned int i = 0; i < N; ++i)
	C[i] = A[i] + B[i];
}
