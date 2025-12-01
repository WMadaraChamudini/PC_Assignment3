// Compile: nvcc -O3 -arch=sm_75 cuda_mat_mul.cu -o cuda_mat_mul
// Run: ./cuda_mat_mul N blockSize <-- N is the size of the matrix,blockSize is the block size

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h> //for runtime API functions

//macro for CUDA error checking
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file, const int line){    
  if (err!=cudaSuccess){        
    fprintf (stderr, "CUDA Error at %s:%d in function %s: %s\n", file, line, func, cudaGetErrorString(err));        
    exit(EXIT_FAILURE);
  }
}

float* alloc_mat(int n){    
  float *m = (float*) malloc((size_t)n*n*sizeof(float));
  if (!m){ 
    perror("malloc"); 
    exit(1); 
  }
  return m;
}

void init_ex(float *A, float *B, int n){    
  for (long i=0; i<(long)n*n; i++){        
    A[i] = (float)((i/n)+(i%n)+1);        
    B[i] = (float)( ((i/n)+1) * ((i%n)+2));    
  }
}

void print_mat(const char *name, float *M, int n){    
  printf("%s:\n", name);
  for (int i=0; i<n; i++){        
    for (int j=0; j<n; j++){ 
      printf("%8.2f ", M[i*n+j]);
    }
    printf("\n");
  }
}

//CUDA kernel for matrix multiplication
__global__ void matmul_kernel(float *A, float *B, float *C, int n){
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row<n && col<n){        
    float sum = 0.0f;
    for (int k=0; k<n; k++){            
      sum += A[row*n+k] * B[k*n+col];
    }
    C[row*n + col]=sum;
  }
}

int main(int argc, char **argv){    
  if (argc<3){
    printf ("Usage: %s N blockSize\n", argv[0]);
    return 1;
  }

  int n = atoi(argv[1]);
  int blockSize = atoi(argv[2]);
  if (n<=0||blockSize<=0){ 
    printf("Invalid arguments\n"); 
    return 1; 
  }

  //host matrices allocation
  float *A = alloc_mat(n);
  float *B = alloc_mat(n);
  float *C = alloc_mat(n);

  init_ex(A,B,n);

  //initialize host C to zeros (to debugging comparison in kernel failures)
  for (long i=0; i<(long)n*n; i++){ 
    C[i] = 0.0f;
  }

  //device memory allocation
  float *d_A, *d_B, *d_C;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_A, n*n*sizeof(float)));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_B, n*n*sizeof(float)));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_C, n*n*sizeof(float)));

  // Copy data -> device
  CHECK_CUDA_ERROR(cudaMemcpy(d_A, A, n*n*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(d_B, B, n*n*sizeof(float), cudaMemcpyHostToDevice));
  //didn't copy host C to device <-- d_C is written to device by kernel

  //configure grid and block
  dim3 dimBlock(blockSize, blockSize);
  dim3 dimGrid( (n+blockSize-1)/blockSize, (n+blockSize-1)/blockSize);

  //CUDA events for timing
  cudaEvent_t start, stop;
  CHECK_CUDA_ERROR(cudaEventCreate(&start));
  CHECK_CUDA_ERROR(cudaEventCreate(&stop));
  CHECK_CUDA_ERROR(cudaEventRecord(start));

  // Launch kernel
  matmul_kernel <<<dimGrid, dimBlock>>> (d_A, d_B, d_C, n);
  CHECK_CUDA_ERROR(cudaGetLastError()); //check for errors from kernel launch

  //wait for kernel to finish
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  CHECK_CUDA_ERROR(cudaEventRecord(stop));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

  float milliseconds = 0;
  CHECK_CUDA_ERROR(cudaEventElapsedTime (&milliseconds, start, stop));
  float seconds = milliseconds/1000.0f; //convert milliseconds to seconds

  //copy result back to host
  CHECK_CUDA_ERROR(cudaMemcpy(C, d_C, n*n*sizeof(float), cudaMemcpyDeviceToHost));

  //print matrices(when N<10)
  if (n<=10){        
    print_mat("Matrix A", A,n); 
    printf("\n");
    print_mat("Matrix B", B,n); 
    printf("\n");
    print_mat("Result Matrix C = A * B", C,n); 
    printf("\n");
  }

  //compute checksum and row sums
  float checksum = 0.0f;
  printf("Summary:\nMatrix size: %d x %d \n", n,n);
  printf("Execution time(seconds): %.6f\n", seconds);
  for(int i=0; i<n; i++){        
    float rowSum = 0.0f;
    for (int j=0; j<n; j++){ 
      rowSum += C[i*n + j];
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
