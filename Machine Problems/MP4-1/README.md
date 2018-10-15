# Machine Problem 4-1:

The objective of this lab is to implement a work-efficient parallel reduction algorithm
on the GPU.

## Running code:
```
make
./vector_reduction
```
\* *Note the make file may need to be modified for use with your environment.*

After the device kernel is invoked, upon completion it will print
out “Test PASSED” to the screen before exiting if successful, otherwise you will see "Test FAILED".

## Arguments
- *No arguemnts*
   -  The application will create a randomly-initialized array
to process. After the device kernel is invoked, it will compute the correct
solution value using the CPU and compare it with the device-computed
solution. If the solutions match (within a certain tolerance), it will print
out “Test PASSED” to the screen before exiting.
- *One argument*
   - The application will initialize the input array with the
values found in the file specified by the argument.

In either mode, the program will print out the final result of the CPU and GPU
computations, and whether or not the comparison passed.

