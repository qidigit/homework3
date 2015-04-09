/******************************************************************************
* FILE: omp_solved3.c
* DESCRIPTION:
*   Run time error
*
*   omp barrier in the function print_results is inside the omp section running by only one thread. It will hold all the processes waiting.
* AUTHOR: Blaise Barney  01/09/04, corrected by Di Qi
* LAST REVISED: 04/07/15
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N     50

int main (int argc, char *argv[]) 
{
int i, nthreads, tid, section;
float a[N], b[N], c[N];
void print_results(float array[N], int tid, int section);

/* Some initializations */
for (i=0; i<N; i++)
  a[i] = b[i] = i * 1.0;

#pragma omp parallel private(c,i,tid,section)
  {
  tid = omp_get_thread_num();
  if (tid == 0)
    {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }

  /*** Use barriers for clean output ***/
  #pragma omp barrier
  printf("Thread %d starting...\n",tid);
  #pragma omp barrier

  #pragma omp sections nowait
    {
    #pragma omp section
      {
      section = 1;
      for (i=0; i<N; i++)
        c[i] = a[i] * b[i];
      print_results(c, tid, section);
      }

    #pragma omp section
      {
      section = 2;
      for (i=0; i<N; i++)
        c[i] = a[i] + b[i];
      print_results(c, tid, section);
      }

    }  /* end of sections */

  /*** Use barrier for clean output ***/
  #pragma omp barrier
  printf("Thread %d exiting...\n",tid);

  }  /* end of parallel section */
  return 0;
}



void print_results(float array[N], int tid, int section) 
{
  int i,j;

  j = 1;
  /*** use critical for clean output ***/
  #pragma omp critical
  {
  printf("\nThread %d did section %d. The results are:\n", tid, section);
  for (i=0; i<N; i++) {
    printf("%e  ",array[i]);
    j++;
    if (j == 6) {
      printf("\n");
      j = 1;
      }
    }
    printf("\n");
  } /*** end of critical ***/

//  #pragma omp barrier
  printf("Thread %d done and synchronized.\n", tid); 

}
  
