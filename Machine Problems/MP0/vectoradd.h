#ifndef _VECTORADD_H_
#define _VECTORADD_H_

// Thread block size
#define BLOCK_SIZE 256

// Vector dimensions
#define VSIZE 256 // vector size

// Vector Structure declaration
typedef struct {
	//length of the vector
    unsigned int length;
	//Pointer to the first element of the vector
    float* elements;
} Vector;


#endif // _VECTORADD_H_

