/******************************************************************************
* FILE: omp_solved4.c
* DESCRIPTION:
*   This very simple program causes a segmentation fault.
*
*   stack size is too small for the double array a with diemsion N^2.
* AUTHOR: Blaise Barney  01/09/04, corrected by Di Qi
* LAST REVISED: 04/07/15
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N 1048

int main (int argc, char *argv[]) 
{
int nthreads, tid, i, j;
// use allocated memory instead
double **a;

/* Fork a team of threads with explicit variable scoping */
#pragma omp parallel shared(nthreads) private(i,j,tid,a)
  {

  /* Obtain/print thread info */
  tid = omp_get_thread_num();
  if (tid == 0) 
    {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  printf("Thread %d starting...\n", tid);

  /* Each thread works on its own private copy of the array */
  a = (double **)malloc(N * sizeof(double *));
  for (i=0; i<N; i++)
      a[i] = (double *)malloc(N * sizeof(double));
  for (i=0; i<N; i++)
    for (j=0; j<N; j++)
      a[i][j] = tid + i + j;

  /* For confirmation */
  printf("Thread %d done. Last element= %f\n",tid,a[N-1][N-1]);

  }  /* All threads join master thread and disband */

  return 0;
}

