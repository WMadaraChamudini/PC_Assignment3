//Compile: gcc -O3 -fopenmp openmp_mat_mul.c -o openmp_mat_mul
//Run: ./openmp_mat_mul N_A_row N_A_col N_B_row N_B_col threads

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

double* alloc_mat(int rows, int cols){
  double *m = (double*) malloc((size_t)rows*cols*sizeof(double));
  if (!m){
    perror("malloc");
    exit(1);
  }
  return m;
}

void init_ex(double *A, double *B, int N_A_row, int N_A_col, int N_B_row, int N_B_col){
  // Initialize Matrix A
  for (long i=0; i<(long)N_A_row*N_A_col; ++i){
    A[i] = (double) ((i/N_A_col)+(i%N_A_col)+1);
  }
  // Initialize Matrix B
  for (long i=0; i<(long)N_B_row*N_B_col; ++i){
    B[i] = (double)(((i/N_B_col)+1) * ((i%N_B_col)+2));
  }
}

void print_mat(const char *name, double *M, int rows, int cols){
  printf("%s:\n", name);
  for (int i=0; i<rows; ++i){
    for (int j=0;j<cols;++j){
      printf("%8.2f ", M [(long) i*cols+j]);
    }
    printf("\n");
  }
}

int main(int argc, char **argv){
  if (argc < 6){
    printf("Usage: %s N_A_row N_A_col N_B_row N_B_col threads\n", argv[0]);
    return 1;
  }
  int N_A_row = atoi(argv[1]);
  int N_A_col = atoi(argv[2]);
  int N_B_row = atoi(argv[3]);
  int N_B_col = atoi(argv[4]);
  int threads = atoi(argv[5]);

  if (N_A_row<=0 || N_A_col<=0 || N_B_row<=0 || N_B_col<=0 || threads<=0){
    printf("Invalid arguments. All dimensions and threads must be positive.\n");
    return 1;
  }

  if (N_A_col != N_B_row){
    printf("Error: N_A_col must be equal to N_B_row for matrix multiplication.\n");
    return 1;
  }

  omp_set_num_threads(threads);

  double *A = alloc_mat(N_A_row, N_A_col);
  double *B = alloc_mat(N_B_row, N_B_col);
  double *C = alloc_mat(N_A_row, N_B_col);  //matrix C will have dimensions N_A_row x N_B_col

  init_ex(A,B, N_A_row, N_A_col, N_B_row, N_B_col);
  for (long i=0; i<(long)N_A_row*N_B_col; ++i){  //Initialize C based on its dimensions
    C[i]=0.0;
  }

  double t0 = omp_get_wtime();

  #pragma omp parallel
  {
    #pragma omp for schedule(static)
    for (int i=0; i<N_A_row; ++i){          // iterate over rows of A
      for (int k=0;k<N_A_col;++k){          // iterate over columns of A (or rows of B)
        double a = A[(long)i*N_A_col + k];
        for (int j=0;j<N_B_col;++j) {       // iterate over columns of B
          C[(long)i*N_B_col + j] += a * B[(long)k*N_B_col + j];
        }
      }
    }
  }

  double t1 = omp_get_wtime();

  //time for execution in seconds
  double Tot_t = t1-t0;

  //print the matrix if N_A_row and N_B_col are smaller than 10
  if (N_A_row<=10 && N_A_col<=10 && N_B_row<=10 && N_B_col<=10){
    print_mat("Matrix A", A, N_A_row, N_A_col);
    printf("\n");
    print_mat("Matrix B", B, N_B_row, N_B_col);
    printf("\n");
    print_mat("Result Matrix C = A * B", C, N_A_row, N_B_col);
    printf("\n");
  }

  double checksum = 0.0;
  printf("Summary:\n");
  printf("Matrix A size: %d x %d\n", N_A_row, N_A_col);
  printf("Matrix B size: %d x %d\n", N_B_row, N_B_col);
  printf("Result Matrix C size: %d x %d\n", N_A_row, N_B_col);
  printf("Threads used: %d\n", threads);
  printf("Execution time: %.6f seconds\n", Tot_t);
  for (int i=0; i<N_A_row; ++i){ // Iterate over rows of C
    double rowSum=0.0;
    for (int j=0;j<N_B_col;++j){ // Iterate over columns of C
      rowSum += C[(long)i*N_B_col+j];
    }
    if (N_A_row <= 50 && N_B_col <= 50) {       // Only print row sums if result matrix dimensions < 50
      printf("Row %2d sum = %.2f\n", i,rowSum);
    }
    checksum+=rowSum;
  }
  printf("Checksum (sum of all elements) = %.6e\n", checksum);

  free(A);
  free(B);
  free(C);
  return 0;
}
