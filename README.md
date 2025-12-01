# PC_Assignment3

## Matrix Multiplication - Parallel Computing Assignment 03

This repository contains implementations of matrix multiplication using different parallel computing paradigms: Serial, OpenMP, MPI, and CUDA.

## Files

| File | Description |
|------|-------------|
| `serial_mat_mul.c` | Serial (single-threaded) matrix multiplication implementation in C |
| `openmp_mat_mul.c` | OpenMP parallelized matrix multiplication using shared memory |
| `mpi_mat_mul.c` | MPI distributed matrix multiplication across multiple processes |
| `cuda_mat_mul.cu` | CUDA GPU-accelerated matrix multiplication |
| `IT23292154_PC_Assignment3.ipynb` | Jupyter notebook demonstrating all implementations |

## Compilation and Usage

### Serial Implementation

```bash
# Compile
gcc -O2 -std=c11 serial_mat_mul.c -o serial_mat_mul

# Run
./serial_mat_mul N_A_row N_A_col N_B_row N_B_col
```

### OpenMP Implementation

```bash
# Compile
gcc -O3 -fopenmp openmp_mat_mul.c -o openmp_mat_mul

# Run
./openmp_mat_mul N_A_row N_A_col N_B_row N_B_col threads
```

### MPI Implementation

```bash
# Compile
mpicc -O3 mpi_mat_mul.c -o mpi_mat_mul

# Run (P is the number of processes)
mpirun --allow-run-as-root --oversubscribe -np P ./mpi_mat_mul N_A_row N_A_col N_B_row N_B_col
```

### CUDA Implementation

```bash
# Compile
nvcc -O3 -arch=sm_75 cuda_mat_mul.cu -o cuda_mat_mul

# Run
./cuda_mat_mul N_A_row N_A_col N_B_row N_B_col blockSize
```

## Parameters

- `N_A_row`: Number of rows in matrix A
- `N_A_col`: Number of columns in matrix A
- `N_B_row`: Number of rows in matrix B
- `N_B_col`: Number of columns in matrix B
- `threads`: Number of OpenMP threads (OpenMP only)
- `blockSize`: CUDA block size (CUDA only)
- `P`: Number of MPI processes (MPI only)

**Note:** For valid matrix multiplication, `N_A_col` must equal `N_B_row`.