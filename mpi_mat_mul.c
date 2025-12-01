// Compile: mpicc -O3 mpi_mat_mul.c -o mpi_mat_mul
// Run: mpirun --allow-run-as-root --oversubscribe -np P ./mpi_mat_mul N_A_row N_A_col N_B_row N_B_col <-- P is the number of processes

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

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
  for (long i=0; i<(long)N_A_row*N_A_col; i++){
    A[i] = (double)( (i/N_A_col)+(i%N_A_col)+1 );
  }
  // Initialize Matrix B
  for (long i=0; i<(long)N_B_row*N_B_col; i++){
    B[i] = (double)( ((i/N_B_col)+1) * ((i%N_B_col)+2) );
  }
}

void print_mat(const char *name, double *M, int rows, int cols){
  printf("%s:\n", name);
  for (int i=0; i<rows; i++){
    for (int j=0; j<cols; j++){
      printf("%8.2f ", M[(long)i*cols+j] );
    }
    printf("\n");
  }
}

int main(int argc, char **argv){
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (argc<5){
    if(rank==0){
      printf("Usage: %s N_A_row N_A_col N_B_row N_B_col\n", argv[0]);
    }
    MPI_Finalize();
    return 1;
  }

  int N_A_row = atoi(argv[1]);
  int N_A_col = atoi(argv[2]);
  int N_B_row = atoi(argv[3]);
  int N_B_col = atoi(argv[4]);

  if (N_A_row<=0 || N_A_col<=0 || N_B_row<=0 || N_B_col<=0){
    if(rank==0){
      printf("Invalid dimensions. All dimensions must be positive.\n");
    }
    MPI_Finalize();
    return 1;
  }

  if (N_A_col != N_B_row){
    if(rank==0){
      printf("Error: N_A_col must be equal to N_B_row for matrix multiplication.\n");
    }
    MPI_Finalize();
    return 1;
  }

  double *A = NULL, *B = NULL, *C = NULL;

  //allocate B on all processes, A and C only on root
  B = alloc_mat(N_B_row, N_B_col);

  if (rank==0){
    A = alloc_mat(N_A_row, N_A_col);
    C = alloc_mat(N_A_row, N_B_col);
    init_ex(A, B, N_A_row, N_A_col, N_B_row, N_B_col);
  }

  //Broadcast B to all processes
  MPI_Bcast(B, N_B_row * N_B_col, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  //Row distribution determination for Matrix A
  int rows_per_proc = N_A_row/size;
  int extra = N_A_row%size;
  int start = rank*rows_per_proc+(rank<extra ? rank:extra);
  int end = start+rows_per_proc+(rank<extra ? 1:0);
  int num_rows = end-start;

  double *A_rows = alloc_mat(num_rows, N_A_col); //local rows of A
  double *C_rows = alloc_mat(num_rows, N_B_col); //local rows of C

  // Scatter rows of A from root to all processes
  // Using MPI_Send/MPI_Recv for custom scatter logic
  if(rank==0){
    for(int r=0; r<size; r++){
      int r_start = r*rows_per_proc+(r<extra ? r:extra);
      int r_end = r_start+rows_per_proc+(r<extra ? 1:0);
      int r_num = r_end-r_start;
      if(r==0){
        for(int i=0; i<r_num*N_A_col; i++){
          A_rows[i] = A[i];
        }
      }else{
        MPI_Send(A + (long)r_start*N_A_col, r_num*N_A_col, MPI_DOUBLE, r, 0, MPI_COMM_WORLD);
      }
    }
  }else{
    MPI_Recv(A_rows, num_rows*N_A_col, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  //start time
  double t0 = MPI_Wtime();

  //compute local C_rows = A_rows*B
  for(int i=0; i<num_rows; i++){           //iterate over local rows of A_rows
    for(int j=0; j<N_B_col; j++){          //iterate over columns of B
      double sum = 0.0;
      for(int k=0; k<N_A_col; k++){        // iterate over columns of A_rows (or rows of B)
        sum += A_rows[(long)i*N_A_col+k] * B[(long)k*N_B_col+j];
      }
      C_rows[(long)i*N_B_col + j] = sum;
    }
  }

  //end time measurement before gathering results
  double t1 = MPI_Wtime();

  //total time
  double Tot_t = t1-t0;

  //gather results from all processes to root
  int *recvcounts = NULL;
  int *displs = NULL;
  if (rank==0){
    recvcounts = (int*) malloc(size * sizeof(int));
    displs = (int*) malloc(size * sizeof(int));
    int offset = 0;
    for(int r=0; r<size; r++){
      int r_start_global = r * rows_per_proc + (r<extra ? r:extra);
      int r_end_global = r_start_global + rows_per_proc + (r<extra ? 1:0);
      recvcounts[r] = (r_end_global - r_start_global) * N_B_col;  // No. of elements to receive from each process
      displs[r] = offset;      //Displacement for each process
      offset += recvcounts[r];
    }
  }

  MPI_Gatherv(C_rows, num_rows*N_B_col, MPI_DOUBLE, C, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  //reduce max execution time to root
  double max_time;
  MPI_Reduce(&Tot_t, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  // Print results on root process
  if(rank==0){
    if(N_A_row<=10 && N_A_col<=10 && N_B_row<=10 && N_B_col<=10){
      print_mat("Matrix A", A, N_A_row, N_A_col);
      printf("\n");
      print_mat("Matrix B", B, N_B_row, N_B_col);
      printf("\n");
      print_mat("Result Matrix C = A*B", C, N_A_row, N_B_col);
      printf("\n");
    }

    double checksum=0.0;
    printf("Summary:\n");
    printf("Matrix A size: %d x %d\n", N_A_row,N_A_col);
    printf("Matrix B size: %d x %d\n", N_B_row,N_B_col);
    printf("Result Matrix C size: %d x %d\n", N_A_row,N_B_col);
    printf("Execution time(seconds): %.6f\n", max_time);
    for(int i=0; i<N_A_row; i++){                 //iterate over rows of C
      double rowSum=0.0;
      for(int j=0; j<N_B_col; j++){               //iterate over columns of C
        rowSum += C[(long)i*N_B_col + j];
      }
      if (N_A_row <= 50 && N_B_col <= 50) { // Only print row sums if result matrix dimensions < 50
        printf(" Row %2d sum = %.2f\n", i,rowSum);
      }
      checksum+=rowSum;
    }
    printf("Checksum (sum of all elements)= %.6e\n", checksum);
  }

  // Free memory
  free(B);      //allocated on all processes
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
