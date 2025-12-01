// Compile: gcc -O3 -fopenmp openmp_mat_mul.c -o openmp_mat_mul
// Run: ./openmp_mat_mul N n <-- N is the size of the matrix, n is the number of threads

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

double* alloc_mat(int n){
  double *m = (double*) malloc((size_t)n*n*sizeof(double));
  if (!m){ 
    perror("malloc"); 
    exit(1); 
  }
  return m;
}

void init_ex(double *A, double *B, int n){    
  for (long i=0; i<(long)n*n; ++i){        
    A[i] = (double) ((i/n)+(i%n)+1);
    B[i] = (double)(((i/n)+1) * ((i%n)+2));
  }
}

void print_mat(const char *name, double *M, int n){    
  printf("%s:\n", name);
  for (int i=0; i<n; ++i){        
    for (int j=0;j<n;++j){
      printf("%8.2f ", M [(long) i*n+j]);
    }
    printf("\n");
  }
}

int main(int argc, char **argv){
  if (argc<3){ 
    printf("Usage: %s N threads\n", argv[0]); 
    return 1; 
  }
  int n = atoi(argv[1]);
  int threads = atoi(argv[2]);
  if (n<=0 || threads<=0){ 
    printf("Invalid args\n"); 
    return 1; 
  }

  omp_set_num_threads(threads);
  double *A = alloc_mat(n), *B = alloc_mat(n), *C = alloc_mat(n);
  init_ex(A,B,n);
  for (long i=0; i<(long)n*n; ++i){
    C[i]=0.0;
  }

 double t0 = omp_get_wtime();
 #pragma omp parallel
 {
  #pragma omp for schedule(static)
    for (int i=0; i<n; ++i){            
      for (int k=0;k<n;++k){                
        double a = A[(long)i*n + k];                
        for (int j=0;j<n;++j) {
          C[(long)i*n + j] += a * B[(long)k*n + j];
        }
      }
    }
  }

  double t1 = omp_get_wtime();
  //time for execution in seconds
  double Tot_t = t1-t0;

  //Print the matrix if n<=10 
  if (n<=10) {        
    print_mat("Matrix A", A,n); 
    printf("\n");
    print_mat("Matrix B", B,n); 
    printf("\n");
    print_mat("Result Matrix C = A * B", C,n); 
    printf("\n");
  }
  double checksum = 0.0;
  printf("Summary:\n");
  printf("Matrix size: %d x %d\n", n, n);
  printf("Threads used: %d\n", threads);
  printf("Execution time: %.6f seconds\n", Tot_t);
  for (int i=0; i<n; ++i){        
    double rowSum=0.0;
    for (int j=0;j<n;++j){
      rowSum += C[(long)i*n+j];
    }
    printf("Row %2d sum = %.2f\n", i,rowSum);        
    checksum+=rowSum;
  }
  printf("Checksum (sum of all elements) = %.6e\n", checksum);

  free(A); free(B); free(C);
  return 0;
}

