PC_Assignment3 – Matrix Multiplication (Serial, OpenMP, MPI, CUDA)

This project implements Matrix Multiplication using four different parallel programming paradigms:

Serial C

OpenMP (Shared Memory Parallelism)

MPI (Distributed Memory Parallelism)

CUDA (GPU Acceleration)

All implementations support dynamic matrix sizes and print execution time, row-wise checksums, and output summaries.

1. Serial Matrix Multiplication
Edit / View Code
vi serial_mat_mul.c

Compile
gcc serial_mat_mul.c -o serial_mat_mul

Run Examples
./serial_mat_mul 4 3 3 5
./serial_mat_mul 40 4000 4000 40

2. OpenMP Matrix Multiplication
Edit / View Code
vi openmp_mat_mul.c

Compile
gcc -O3 -fopenmp openmp_mat_mul.c -o openmp_mat_mul

Run Examples

Format:

./openmp_mat_mul A_rows A_cols B_rows B_cols threads

./openmp_mat_mul 4 3 3 5 4
./openmp_mat_mul 40 4000 4000 40 2
./openmp_mat_mul 40 4000 4000 40 4

3. MPI Matrix Multiplication
Edit / View Code
vi mpi_mat_mul.c

Compile
mpicc -O3 mpi_mat_mul.c -o mpi_mat_mul

Run Examples

Format:

mpirun -np <processes> ./mpi_mat_mul A_rows A_cols B_rows B_cols

mpirun --allow-run-as-root --oversubscribe -np 4 ./mpi_mat_mul 4 3 3 5
mpirun --allow-run-as-root --oversubscribe -np 2 ./mpi_mat_mul 40 4000 4000 40
mpirun --allow-run-as-root --oversubscribe -np 4 ./mpi_mat_mul 40 4000 4000 40

4. CUDA Matrix Multiplication
Edit / View Code
vi cuda_mat_mul.cu

Compile
nvcc -O3 -arch=sm_75 cuda_mat_mul.cu -o cuda_mat_mul

Run Examples

Format:

./cuda_mat_mul A_rows A_cols B_rows B_cols blockSize

./cuda_mat_mul 4 3 3 5 16
./cuda_mat_mul 1024 1024 1024 1024 32

Performance Evaluation

This project allows detailed performance analysis, including:

Execution time

Row-wise checksum validation

Speedup calculation

OpenMP multi-thread comparison

MPI multi-process comparison

CUDA block-size evaluation

Required Graphs (for the Report)

Threads vs Execution Time (OpenMP)

Processes vs Speedup (MPI)

Block Size vs Execution Time (CUDA)

Serial vs OpenMP vs MPI vs CUDA (overall comparison)

Requirements
Install the following tools:
GCC
sudo apt install build-essential

OpenMP

Included with GCC.

MPI (MPICH / OpenMPI)
sudo apt install mpich

CUDA Toolkit

Download from NVIDIA website (Linux/Windows/WSL2).
Or use Google Colab for GPU execution.
