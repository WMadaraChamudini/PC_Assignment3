// Compile: mpicc -O3 mpi_mat_mul.c -o mpi_mat_mul
// Run: mpirun --allow-run-as-root --oversubscribe -np P ./mpi_mat_mul N <-- P is the number of processes, N is the size of the matrix

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

double* alloc_mat(int n){
  double *m = (double*) malloc((size_t)n*n*sizeof(double));
  if (!m){ 
    perror("malloc"); 
    exit(1); 
  }
  return m;
}

void init_ex(double *A, double *B, int n){
  for (long i=0; i<(long)n*n; i++){
    A[i] = (double)( (i/n)+(i%n)+1 );
    B[i] = (double)( ((i/n)+1) * ((i%n)+2) );
  }
}

void print_mat(const char *name, double *M, int n){    
  printf("%s:\n", name);
  for (int i=0; i<n; i++){
    for (int j=0; j<n; j++){
      printf("%8.2f ", M[i*n+j] );
    }
    printf("\n");
  }
}

int main(int argc, char **argv){    
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (argc<2){        
    if(rank==0){
      printf("Usage: %s N\n", argv[0]);
    }
    MPI_Finalize(); 
    return 1;
  }

  int n = atoi(argv[1]);
  if (n<=0){ 
    if(rank==0){ 
      printf("Invalid N\n"); 
    }
    MPI_Finalize(); 
    return 1; 
  }

  double *A = NULL, *B = alloc_mat(n), *C = NULL;    

  if (rank==0){
    A = alloc_mat(n);
    C = alloc_mat(n);
    init_ex(A,B,n);
  }

  //broadcast B to all processes
  MPI_Bcast(B, n*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  //row distribution determination
  int rows_per_proc = n/size;
  int extra = n%size;
  int start = rank*rows_per_proc + (rank<extra ? rank:extra);
  int end = start + rows_per_proc + (rank<extra ? 1:0);
  int num_rows = end-start;

  double *A_rows = (double*) malloc(num_rows * n * sizeof(double));
  double *C_rows = (double*) malloc(num_rows * n * sizeof(double));

  // scatter rows of A    
  if(rank==0){
    for(int r=0; r<size; r++){
      int r_start = r*rows_per_proc + (r<extra?r:extra);
      int r_end = r_start + rows_per_proc + (r<extra?1:0);
      int r_num = r_end - r_start;
      if(r==0){
        for(int i=0; i<r_num*n; i++){ 
          A_rows[i] = A[i];
        }
      }else{
        MPI_Send(A + r_start*n, r_num*n, MPI_DOUBLE, r, 0, MPI_COMM_WORLD);
      }
    }
  }else{
    MPI_Recv(A_rows, num_rows*n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  //start time
  double t0 = MPI_Wtime();

  //compute local C
  for(int i=0; i<num_rows; i++){        
    for(int j=0; j<n; j++){           
      double sum = 0.0;
      for(int k=0; k<n; k++){
        sum += A_rows[i*n+k] * B[k*n+j];            
      }
      C_rows[i*n + j] = sum;
    }
  }

  //end time
  double t1 = MPI_Wtime();

  //Total time
  double Tot_t = t1-t0;

  //gather results
  int *recvcounts = NULL;
  int *displs = NULL;
  if (rank==0){        
    recvcounts = (int*) malloc(size * sizeof(int));       
    displs = (int*) malloc(size * sizeof(int));
    int offset = 0;
    for(int r=0; r<size; r++){
      int r_start = r*rows_per_proc + (r<extra ? r:extra);
      int r_end = r_start + rows_per_proc + (r<extra ? 1:0);
      recvcounts[r] = (r_end-r_start)*n;            
      displs[r] = offset;
      offset += recvcounts[r];
    }
  }

  MPI_Gatherv(C_rows, num_rows*n, MPI_DOUBLE, C, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // reduce max execution time to root
  double max_time;
  MPI_Reduce(&Tot_t, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  // Print results on root
  if(rank==0){
    if(n<=10){            
      print_mat("Matrix A", A,n); 
      printf("\n");
      print_mat("Matrix B", B,n); 
      printf("\n");
      print_mat("Result Matrix C = A*B", C,n); 
      printf("\n");
    }

    double checksum=0.0;
    printf("Summary:\n Matrix size: %d x %d\n", n,n);
    printf("Execution time(seconds): %.6f\n", max_time);
    for(int i=0; i<n; i++){
      double rowSum=0.0;
      for(int j=0; j<n; j++){ 
        rowSum += C[i*n + j];
      }
      printf(" Row %2d sum = %.2f\n", i,rowSum);            
      checksum += rowSum;
    }
    printf("Checksum(sum of all elements)= %.6e\n", checksum);
  }

  free(B); 
  free(A_rows); 
  free(C_rows);
  if(rank==0){ 
    free(A); 
    free(C); 
    free(recvcounts); 
    free(displs); 
  }

  MPI_Finalize();
  return 0;
}
