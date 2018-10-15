#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>


#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define TILE_SIZE 1024
// You can use any other block size you wish.
#define BLOCK_SIZE 256


// Host Helper Functions (allocate your own data structure...)



// Device Functions



// Kernel Functions



// **===-------- Modify the body of this function -----------===**
// You may need to make multiple kernel calls. Make your own kernel
// functions in this file, and then call them from here.
// Note that the code has been modified to ensure numElements is a multiple 
// of TILE_SIZE
void prescanArray(unsigned int *outArray, unsigned int *inArray, int numElements)
{



}
// **===-----------------------------------------------------------===**


#endif // _PRESCAN_CU_
