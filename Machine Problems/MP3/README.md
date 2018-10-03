# Machine Problem 3:

The objective of this lab is to learn how constant memory and shared memory
can be used to alleviate memory bandwidth bottlenecks in the context of a
convolution computation. This MP is a tiled implementation of a matrix convolution.
This assignment will have a constant 5x5 convolution kernel, but will
have arbitrarily sized “images”.

Matrix convolution is primarily used in image processing for tasks such as image
enhancing, blurring, etc. A standard image convolution formula for a 5x5
convolution filter M with an Image N is:

![alt text](https://latex.codecogs.com/gif.latex?P%28i%2C%20j%29%20%3D%20%5Csum_%7Ba%3D0%7D%5E%7B4%7D%5Csum_%7Bb%3D0%7D%5E%7B4%7DM%5Ba%5D%5Bb%5D*N%5Bi&plus;a-2%5D%5Bj&plus;b-2%5D%2C%20%5C%5C%5C%5Cwhere%5C%2C0%5Cleqslant%20i%3C%20N.height%5C%2Cand%5C%2C0%5Cleq%20j%3C%20N.width)

For this MP, elements that are “outside” Matrix N are treated as if they had
the value zero.

## Running code:
```
make
./2Dconvolution
```
\* *Note the make file may need to be modified for use with your environment.*

After the device convolution is invoked, upon completion it will print
out “Test PASSED” to the screen before exiting if successful, otherwise you will see "Test FAILED".

## Arguments
- *No arguemnts*
   - The application will create a randomized Filter M and
Image N. A CPU implementation of the convolution algorithm will be
used to generate a correct solution which will be compared with your
program’s output. If it matches (within a certain tolerance), it will print
out “Test PASSED” to the screen before exiting.
- *One argument*
   - The application will create a randomized Filter M and
Image N, and write the device-computed output to the file specified by
the argument.
- *Three arguments*
   - The application will read the filter and image from
provided files. The first argument should be a file containing two integers
representing the image height and width respectively. The second and
third function arguments should be files which have exactly enough entries
to fill the Filter M and Image N respectively. No output is written to file
- *Four arguments* 
   - The application will read its inputs using the files provided
by the first three arguments, and write its output to the file provided
in the fourth.


