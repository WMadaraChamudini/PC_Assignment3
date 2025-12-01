//compile: gcc -O2 -std=c11 serial_mat_mul.c -o serial_mat_mul
//run: ./serial_mat_mul N  <-- N is the size of the matrix

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double* alloc_mat(int n){
  double *m =(double*) malloc((size_t) n*n*sizeof(double));
  if (!m){ 
    perror("malloc"); 
    exit(1); 
  }
  return m;
}

void init_ex(double *A,double *B,int n){
  for (long i=0; i<(long)n*n; ++i){
    A[i]=(double)((i/n) + (i%n)+1);           
    B[i]=(double)( ((i/n)+1) * ((i % n)+2) );  
  }
}

void print_mat(const char *name, double *M, int n){
    printf("%s:\n", name);
    for (int i=0;i<n;++i) {
        for (int j=0;j<n;++j) {
            printf( "%8.2f ", M[(long)i*n + j] );
        }
        printf("\n");
    }
}

int main(int argc, char **argv){
  int n=3;

  if (argc>=2){
    n = atoi(argv[1]);
  }

  if (n<=0){ 
    printf("Invalid N\n"); 
    return 1; 
  }

  double *A = alloc_mat(n);
  double *B = alloc_mat(n);
  double *C = alloc_mat(n);

  init_ex(A,B,n);
  for (long i=0; i<(long)n*n; ++i){
    C[i]=0.0;
  }

  clock_t t0 = clock();

  for (int i=0;i<n;++i) {
    for (int j=0;j<n;++j) {
      double sum = 0.0;
      for (int k=0;k<n;++k) sum += A[(long)i*n + k] * B[(long)k*n + j];
      C[(long)i*n + j] = sum;
    }
  }
  
  clock_t t1 = clock();

  //time for execution in seconds
  double Tot_t=(double) (t1-t0)/CLOCKS_PER_SEC;

  //Print the matrix if n<=10
  if (n<=10) {
    print_mat("Matrix A", A,n); 
    printf("\n");
    print_mat("Matrix B", B,n); 
    printf("\n");
    print_mat("Result Matrix C = A * B", C,n); 
    printf("\n");
  }

  // checksum and per-row sums
  double checksum = 0.0;
  printf("Summary:\n");
  printf("Matrix size: %d x %d\n", n,n);
  printf("Execution time: %.6f seconds\n", Tot_t);
  for (int i=0;i<n;++i){
    double rowSum = 0.0;        
    for (int j=0;j<n;++j){
      rowSum += C[(long)i*n + j];
    }
    printf(" Row %2d sum = %.2f\n", i, rowSum);
    checksum += rowSum;
  }
  printf("Checksum (sum of all elements) = %.6e\n", checksum);

  free(A); 
  free(B); 
  free(C);
  return 0;
}

