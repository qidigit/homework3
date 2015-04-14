# homework3

I run jacobi-omp and gs-omp on Stampede using 'largemem' with at most 32 threads. The tests are run in scale N=1E9 with 10 iterations. Below are the results for jacobian and Gauss-Seidel iteration:

number of threads      32      16       8         4        2        1

GS (s)              5.585   5.270   9.885    19.656   39.257   78.466
Jacobi (s)          3.655   4.962   9.782    19.476   38.987   77.845
