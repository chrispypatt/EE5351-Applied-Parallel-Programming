#include <stdio.h>
#include <math.h>
//#include <float.h>

////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C" 
void computeGold( unsigned int* reference, unsigned int* idata, const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! Each element is the sum of the elements before it in the array.
//! @param reference  reference data, computed but preallocated
//! @param idata      input data as provided to device
//! @param len        number of elements in reference / idata
////////////////////////////////////////////////////////////////////////////////
void
computeGold( unsigned int* reference, unsigned int* idata, const unsigned int len) 
{
  reference[0] = 0;
  unsigned int total_sum = 0;
  unsigned int i;
  for( i = 0; i < len; ++i) 
  {
      total_sum += idata[i];
  }
  *reference = total_sum;
}

