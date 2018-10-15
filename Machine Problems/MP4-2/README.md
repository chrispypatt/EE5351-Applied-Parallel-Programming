# Machine Problem 4-2:

This MP is an implementation of a parallel prefix sum. The algorithm is also called scan, and will be referred to as such in this description. Scan is a useful building block for many parallel algorithms, such as radix sort, quicksort, tree operations, and histograms. Exclusive scan applied to an Array A will produce an Array A’, where:

![](https://latex.codecogs.com/gif.latex?%5Ccenter%20%7BA%7D%27%5Bi%5D%20%3D%20%7BA%7D%27%5Bi%20-%201%5D%20&plus;%20A%5Bi%20-1%5D%20%3A%20A%5B0%5D%20%3D%200%20%5C%5C%20%5Ccenter%20Or%3A%20%5C%5C%20%5Ccenter%20%7BA%7D%27%5Bi%5D%20%3D%20%5Csum_%7Bj%3D0%7D%5E%7Bi-1%7D%20A%5Bj%5D%3AA%5B0%5D%20%3D%200)


## Running code:
```
make
./scan_largearray
```
\* *Note the make file may need to be modified for use with your environment.*

## Arguments
- *No arguemnts*
   -  Randomly generate input data and compare the GPU’s result against the host’s result.
- *One argument*
   - Randomly generate input data and write the result to the file specified by the argument.
- *Two arguments*
   -  The first argument specifies a file that contains the array size. Randomly generate input data and write it to the file specified by the second argument. (This mode is good for generating test arrays.)
- *Three arguments*
   -  The first argument specifies a file that contains the array size. The second and third arguments specify the input file and output file, respectively.

