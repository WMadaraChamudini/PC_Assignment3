# PC_Assignment3 - Matrix Multiplication (Serial, OpenMP, MPI, CUDA)
This project implements Matrix Multiplication using four different programming paradigms:
**      Serial C

      OpenMP (Shared Memory Parallelism)

      MPI (Distributed Memory Parallelism)

      CUDA (GPU Acceleration)
**
All versions support dynamic matrix sizes, print output summaries, row-wise checksums, and execution times.


## 1. Serial Matrix Multiplication
Edit/view code: vi serial_mat_mul.c
Compile: gcc serial_mat_mul.c -o serial_mat_mul
Run examples:
      ./serial_mat_mul 4 3 3 5
      ./serial_mat_mul 40 4000 4000 40

## 2. OpenMP Matrix Multiplication
Edit / view code: vi openmp_mat_mul.c
Compile using OpenMP: gcc -O3 -fopenmp openmp_mat_mul.c -o openmp_mat_mul
Run examples: Format: ./openmp_mat_mul A_rows A_cols B_rows B_cols threads
        ./openmp_mat_mul 4 3 3 5 4
        ./openmp_mat_mul 40 4000 4000 40 2
        ./openmp_mat_mul 40 4000 4000 40 4

## 3. MPI Matrix Multiplication
Edit / view code: vi mpi_mat_mul.c
Compile using MPICC: mpicc -O3 mpi_mat_mul.c -o mpi_mat_mul
Run examples: Format: mpirun -np P ./mpi_mat_mul A_rows A_cols B_rows B_cols
        mpirun --allow-run-as-root --oversubscribe -np 4 ./mpi_mat_mul 4 3 3 5
        mpirun --allow-run-as-root --oversubscribe -np 2 ./mpi_mat_mul 40 4000 4000 40
        mpirun --allow-run-as-root --oversubscribe -np 4 ./mpi_mat_mul 40 4000 4000 40

## 4. CUDA Matrix Multiplication
Edit / view code: vi cuda_mat_mul.cu
Compile using NVCC: nvcc -O3 -arch=sm_75 cuda_mat_mul.cu -o cuda_mat_mul
Run examples: Format: ./cuda_mat_mul A_rows A_cols B_rows B_cols blockSize
        ./cuda_mat_mul 4 3 3 5 16
        ./cuda_mat_mul 1024 1024 1024 1024 32

## Performance Evaluation
The project supports analysis for:
                    - Execution time
                    - Row-wise checksum validation
                    - Speedup calculation
                    - Multi-thread comparison (OpenMP)
                    - Multi-process comparison (MPI)
                    - GPU block-size evaluation (CUDA)

## Graphs (In report):
        - Threads vs Execution Time (OpenMP)
        - Processes vs Speedup (MPI)
        - Block Size vs Time (CUDA)
        - Overall Serial vs OpenMP vs MPI vs CUDA comparison

## Requirements
Ensure you have:
    GCC: sudo apt install build-essential
    OpenMP (built into GCC)
    MPI (OpenMPI/MPICH): sudo apt install mpich
    CUDA Toolkit (for NVIDIA GPUs): Install from NVIDIA website or WSL2 instructions.(or use Google Colab)
