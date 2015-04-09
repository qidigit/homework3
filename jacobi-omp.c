#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define EPS 1E-6
#define MAX_ITER 1E1


double residual(int N, double *f, double *u);

// Jacobi iteration step
void Jacobi_iter(int N, double *f, double *u)
{
  double *u_temp;
  double hinv2 = (N+1)*(N+1);

  // boundary condition
  u[0]=0; u[N+1]=0;

  u_temp = (double *) malloc(sizeof(double)*(N + 1));
  int i;
  #pragma omp parallel for private(i)
  for(i = 1; i <= N; i++) {
      u_temp[i] = (f[i]+hinv2*u[i-1]+hinv2*u[i+1]) / (2*hinv2);
  }

  #pragma omp parallel for private(i)
  for(i = 1; i <= N; i++) {
      u[i] = u_temp[i];
  }

  free(u_temp);
}


// calculate the relative error
double error(int N, double *u, double *v)
{
  double a = 0, b = 0;
  int i;

  #pragma omp parallel for reduction(+: a, b)
  for(i = 1; i <= N; i++) {
      a += (u[i]-v[i])*(u[i]-v[i]);
      b += u[i]*u[i];
  }

  return sqrt(a)/sqrt(b);
}

// calculate the residual
double residual(int N, double *f, double *u)
{
   double resi = 0;
   double hinv2 = (N+1)*(N+1);

   int i;
   #pragma omp parallel for reduction(+:resi)
   for(i = 1; i <= N; i++) {
       double diff = u[i]*2*hinv2-u[i-1]*hinv2-u[i+1]*hinv2-f[i];
       resi += diff*diff;
   }

   return sqrt(resi);
}

int main(int argc, char **argv)
{
  long iter = 0;
//  double err, resi, resi0;
  double *f, *u_J, *u_truth;

  if (argc != 2) {
      fprintf(stderr, "Function needs vector size as input arguments!\n");
      abort(); 
  } 
  long N = atol(argv[1]);


  f = (double *) malloc(sizeof(double)*(N+2));
  u_J = (double *) malloc(sizeof(double)*(N+2));

  // get truth & initial value
  double x;
  u_truth = (double *) malloc(sizeof(double)*(N+2));
  int i, tid, nthreads;
  
#pragma omp parallel private(tid)
  {
  tid = omp_get_thread_num();
  if (tid == 0)
      nthreads = omp_get_num_threads();
  printf("Jacobi iteration of thread %d started...\n", tid);
  }
  
  #pragma omp parallel for private(i, x)
  for(i = 0; i <= N+1; i++) {
      x = (double)i / (N+1);
      u_truth[i] = .5 * x * (1-x);

      f[i]=1; u_J[i]=0;
  }

  // get residual for initial value 
//  resi = residual(N, f, u_J);
//  resi0 = resi;
 
  double tic, toc, elapsed; 
  // Jacobi
  tic = omp_get_wtime();
  while(iter < MAX_ITER) {
  	Jacobi_iter(N, f, u_J);
//        resi = residual(N, f, u_J);
        iter++;
  }
  toc = omp_get_wtime();
  elapsed = toc-tic;

//  err = error(N, u_truth, u_J);

  // print results
  printf("\nThreads = %d: take %ld iterations, run_time = %f seconds.\n", nthreads, iter, elapsed);

  // pointwise value
//  for(i=0; i <= N+1; i++) {
//     if(i%(N/10) == 0) {
//        printf("x=%f: u_truth=%.10f, u_J=%.10f\n", (double)i/(N+1), u_truth[i], u_J[i]);
//     }
//  }

  
  free(u_J);
  free(u_truth);
  free(f);

  return 0;
}
