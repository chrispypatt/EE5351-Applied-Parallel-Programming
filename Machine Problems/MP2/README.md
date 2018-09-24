# Machine Problem 2:

A tiled matrix multiplication program. The purpose of this lab is to learn about shared memory tiling and apply it to a matrix multiplication problem to alleviate the memory bandwidth bottleneck.

## Running code:
```
make
./matrixmul
```
\* *Note the make file may need to be modified for use with your environment.*

After the device multiplication is invoked, it will compute the correct
solution matricies using the CPU, and compare that solution with the device computed solution. If it matches (within a certain tolerance), it will print
out “Test PASSED” to the screen before exiting, otherwise you will see "Test FAILED".

## Arguments
- *No arguemnts*
   - The application will create two randomly sized and initialized
matrices such that the matrix operation M * N is valid, and P
is properly sized to hold the result. After the device multiplication is invoked,
it will compute the correct solution matrix using the CPU, and
compare that solution with the device-computed solution. If it matches
(within a certain tolerance), if will print out “Test PASSED” to the screen
before exiting. 
- *One argument*
   - The application will use the random initialization to create
the input matrices, and write the device-computed output to the file
specified by the argument.
- *Three arguments*
   - The application will read the input matrices from provided
files. The first argument should be a file containing three integers.
The first, second, and third integers will be used as M.height, M.width,
and N.width respectively . The second and third arguments will be expected
to be files which have exactly enough entries to fill matrices M and
N respectively. No output is written to file.
- *Four arguments* 
   - The application will read its inputs from the files provided
by the first three arguments, as described above, and write its output
to the file provided in the fourth.


