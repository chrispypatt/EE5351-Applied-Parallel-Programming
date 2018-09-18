# Machine Problem 0:

"Hello World" of parallel programming. Modify code to perform vector addition using CUDA C. The purpose of this machine problem is to get familiar with writing, compiling and running CUDA programs

## Running code:
```
make

./vectoradd
```

After the device addition is invoked, it will compute the correct
solution vector using the CPU, and compare that solution with the device computed
solution. If it matches (within a certain tolerance), it will print
out “Test PASSED” to the screen before exiting, otherwise you will see "Test FAILED".

## Arguments
- *No arguemnts*
   - the application will create two randomly initialized vectors to add. 
- *One argument*
   - the application will use the random initialization to create the input vectors, and write the device-computed output to the file specified by the argument.
- *Two arguments*
   - the application will initialize the two input vectors with the values found in the files provided as arguments. No output is written to file.
- *Three arguments* 
   - the application will read its inputs from the files provided by the first two arguments, and write its output to the file provided in the third.
