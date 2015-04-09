/******************************************************************************
* FILE: omp_solved2.c
* DESCRIPTION:
*   Another OpenMP program with a bug.
*
*   tid defined outside the parallel regime is a shared variable, so it will be overwritten by the last evaluation in the parallel regime. 
*   and it is unsafe  to use schedule in the omp for loop since the conflict in access
* AUTHOR: Blaise Barney, corrected by Di Qi
* LAST REVISED: 04/07/15 
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[]) 
{
int nthreads, i, tid;
double total;

/*** Spawn parallel region ***/
// set i, tid private variable
#pragma omp parallel private(i, tid) 
  {
  /* Obtain thread number */
  tid = omp_get_thread_num();
  /* Only master thread does this */
  if (tid == 0) {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  printf("Thread %d is starting...\n",tid);

  #pragma omp barrier

  /* do some work */
  total = 0.0;
  // conflict by access in the summation
  #pragma omp for reduction(+:total) //schedule(dynamic,10)
  for (i=0; i<1000000; i++) 
     total = total + i*1.0;

  printf ("Thread %d is done! Total= %e\n",tid,total);

  } /*** End of parallel region ***/

  return 0;
}
