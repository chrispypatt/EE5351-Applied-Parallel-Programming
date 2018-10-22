# Machine Problem 5: Histogramming

Histograms are a commonly used analysis tool in image processing and data
mining applications. They show the frequency of occurrence of data elements
over discrete intervals, also known as bins. A simple example for the use of
histograms is determining the distribution of a set of grades.

Example:
Grades: 0, 1, 1, 4, 0, 2, 5, 5, 5
The above grades ranging from 0 to 5 result in the following 6-bin histogram:
Histogram: 2, 2, 1, 0, 1, 3

The task is to implement an optimized function: void opt_2dhisto(...)

## Arguments
- *No arguemnts*
   -  The application will use a default seed value for the random
number generator when creating the input image.
- *One argument*
   - The application will use the seed value provided as a
command-line argument. When measuring the performance of your application,
we will use this mode with a set of different seed values.

When run, the application will report the timing information for the
sequential code followed by the timing information of the parallel implementation.
It will also compare the two outputs and print “Test PASSED” if they are
identical, or “Test FAILED” otherwise.

