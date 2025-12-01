// Compile: nvcc -O3 -arch=sm_75 cuda_mat_mul.cu -o cuda_mat_mul
// Run: ./cuda_mat_mul N_A_row N_A_col N_B_row N_B_col blockSize

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>      //for runtime API functions

//macro for CUDA error checking
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file, const int line){
  if (err!=cudaSuccess){
    fprintf (stderr, "CUDA Error at %s:%d in function %s: %s\n", file, line, func, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

float* alloc_mat(int rows, int cols){
  float *m = (float*) malloc((size_t)rows*cols*sizeof(float));
  if (!m){
    perror("malloc");
    exit(1);
  }
  return m;
}

void init_ex(float *A, float *B, int N_A_row, int N_A_col, int N_B_row, int N_B_col){
  //initialize Matrix A
  for (long i=0; i<(long)N_A_row*N_A_col; i++){
    A[i] = (float)((i/N_A_col)+(i%N_A_col)+1);
  }
  //initialize Matrix B
  for (long i=0; i<(long)N_B_row*N_B_col; i++){
    B[i] = (float)( ((i/N_B_col)+1) * ((i%N_B_col)+2));
  }
}

void print_mat(const char *name, float *M, int rows, int cols){
  printf("%s:\n", name);
  for (int i=0; i<rows; i++){
    for (int j=0; j<cols; j++){
      printf("%8.2f ", M[(long)i*cols+j]);
    }
    printf("\n");
  }
}

//CUDA kernel for matrix multiplication
__global__ void matmul_kernel(float *A, float *B, float *C, int N_A_row, int N_A_col, int N_B_row, int N_B_col){
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < N_A_row && col < N_B_col){
    float sum = 0.0f;
    for (int k=0; k<N_A_col; k++){
      sum += A[(long)row*N_A_col+k] * B[(long)k*N_B_col+col];
    }
    C[(long)row*N_B_col + col]=sum;
  }
}

int main(int argc, char **argv){
  if (argc<6){
    printf ("Usage: %s N_A_row N_A_col N_B_row N_B_col blockSize\n", argv[0]);
    return 1;
  }

  int N_A_row = atoi(argv[1]);
  int N_A_col = atoi(argv[2]);
  int N_B_row = atoi(argv[3]);
  int N_B_col = atoi(argv[4]);
  int blockSize = atoi(argv[5]);

  if (N_A_row<=0 || N_A_col<=0 || N_B_row<=0 || N_B_col<=0 || blockSize<=0){
    printf("Invalid arguments. All dimensions and block size must be positive.\n");
    return 1;
  }

  if (N_A_col != N_B_row){
    printf("Error: N_A_col must be equal to N_B_row for matrix multiplication.\n");
    return 1;
  }

  //host matrices allocation
  float *A = alloc_mat(N_A_row, N_A_col);
  float *B = alloc_mat(N_B_row, N_B_col);
  float *C = alloc_mat(N_A_row, N_B_col);   //Result matrix C will have dimensions N_A_row x N_B_col

  init_ex(A,B,N_A_row, N_A_col, N_B_row, N_B_col);

  //initialize host C to zeros(to debugging comparison in kernel failures)
  for (long i=0; i<(long)N_A_row*N_B_col; i++){
    C[i] = 0.0f;
  }

  //device memory allocation
  float *d_A, *d_B, *d_C;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_A, (size_t)N_A_row*N_A_col*sizeof(float)));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_B, (size_t)N_B_row*N_B_col*sizeof(float)));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_C, (size_t)N_A_row*N_B_col*sizeof(float)));

  // Copy data -> device
  CHECK_CUDA_ERROR(cudaMemcpy(d_A, A, (size_t)N_A_row*N_A_col*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(d_B, B, (size_t)N_B_row*N_B_col*sizeof(float), cudaMemcpyHostToDevice));
  //didn't copy host C to device <-- d_C is written to device by kernel

  //configure grid and block
  dim3 dimBlock(blockSize, blockSize);
  dim3 dimGrid( (N_B_col+blockSize-1)/blockSize, (N_A_row+blockSize-1)/blockSize);   //grid dimensions based on C (N_A_row x N_B_col)

  //CUDA events for timing
  cudaEvent_t start, stop;
  CHECK_CUDA_ERROR(cudaEventCreate(&start));
  CHECK_CUDA_ERROR(cudaEventCreate(&stop));
  CHECK_CUDA_ERROR(cudaEventRecord(start));

  // Launch kernel
  matmul_kernel <<<dimGrid, dimBlock>>> (d_A, d_B, d_C, N_A_row, N_A_col, N_B_row, N_B_col);
  CHECK_CUDA_ERROR(cudaGetLastError());   //check for errors from kernel launch

  //wait for kernel to finish
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  CHECK_CUDA_ERROR(cudaEventRecord(stop));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

  float milliseconds = 0;
  CHECK_CUDA_ERROR(cudaEventElapsedTime (&milliseconds, start, stop));
  float seconds = milliseconds/1000.0f; //convert milliseconds to seconds

  //copy result back to host
  CHECK_CUDA_ERROR(cudaMemcpy(C, d_C, (size_t)N_A_row*N_B_col*sizeof(float), cudaMemcpyDeviceToHost));

  //print matrices(when N_A_row and N_B_col are smaller than 10)
  if (N_A_row<=10 && N_A_col<=10 && N_B_row<=10 && N_B_col<=10){
    print_mat("Matrix A", A,N_A_row, N_A_col);
    printf("\n");
    print_mat("Matrix B", B,N_B_row, N_B_col);
    printf("\n");
    print_mat("Result Matrix C = A * B", C,N_A_row, N_B_col);
    printf("\n");
  }

  //compute checksum and row sums
  float checksum = 0.0f;
  printf("Summary:\n");
  printf("Matrix A size: %d x %d \n", N_A_row, N_A_col);
  printf("Matrix B size: %d x %d \n", N_B_row, N_B_col);
  printf("Result Matrix C size: %d x %d \n", N_A_row, N_B_col);
  printf("Execution time(seconds): %.6f\n", seconds);
  for(int i=0; i<N_A_row; i++){                //iterate over rows of C
    float rowSum = 0.0f;
    for (int j=0; j<N_B_col; j++){             //iterate over columns of C
      rowSum += C[(long)i*N_B_col + j];
    }
    printf(" Row %2d sum = %.2f\n", i,rowSum);
    checksum += rowSum;
  }
  printf("Checksum (sum of all elements) = %.6e\n", checksum);

  //free memory
  free(A);
  free(B);
  free(C);
  CHECK_CUDA_ERROR(cudaFree(d_A));
  CHECK_CUDA_ERROR(cudaFree(d_B));
  CHECK_CUDA_ERROR(cudaFree(d_C));
  CHECK_CUDA_ERROR(cudaEventDestroy(start));
  CHECK_CUDA_ERROR(cudaEventDestroy(stop));

  return 0;
}
