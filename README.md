# PC_Assignment3 - Matrix Multiplication (Serial, OpenMP, MPI, CUDA)
This project implements **Matrix Multiplication** using four different programming paradigms:
* **Serial C**
* **OpenMP** (Shared Memory Parallelism)
* **MPI** (Distributed Memory Parallelism)
* **CUDA** (GPU Acceleration)

All implementations support dynamic matrix sizes, execution-time measurement, row-wise checksums and output summaries.

---

## 1. Serial Matrix Multiplication
### Edit/View Code
```bash
vi serial_mat_mul.c
````

### Compile

```bash
gcc serial_mat_mul.c -o serial_mat_mul
```

### Run Examples

```bash
./serial_mat_mul 4 3 3 5
./serial_mat_mul 40 4000 4000 40
./serial_mat_mul 4000 400 400 4000
```

---

## 2. OpenMP Matrix Multiplication

### Edit/View Code

```bash
vi openmp_mat_mul.c
```

### Compile

```bash
gcc -O3 -fopenmp openmp_mat_mul.c -o openmp_mat_mul
```

### Run Examples

**Format:**

```bash
./openmp_mat_mul A_rows A_cols B_rows B_cols threads
```

```bash
./openmp_mat_mul 4 3 3 5 4
./openmp_mat_mul 40 4000 4000 40 2
./openmp_mat_mul 40 4000 4000 40 4
./openmp_mat_mul 4000 400 400 4000 1
./openmp_mat_mul 4000 400 400 4000 2
./openmp_mat_mul 4000 400 400 4000 4
```

---

## 3. MPI Matrix Multiplication

### Edit/View Code

```bash
vi mpi_mat_mul.c
```

### Compile

```bash
mpicc -O3 mpi_mat_mul.c -o mpi_mat_mul
```

### Run Examples

**Format:**

```bash
mpirun -np <processes> ./mpi_mat_mul A_rows A_cols B_rows B_cols
```

```bash
mpirun --allow-run-as-root --oversubscribe -np 4 ./mpi_mat_mul 4 3 3 5
mpirun --allow-run-as-root --oversubscribe -np 2 ./mpi_mat_mul 40 4000 4000 40
mpirun --allow-run-as-root --oversubscribe -np 4 ./mpi_mat_mul 40 4000 4000 40
mpirun --allow-run-as-root --oversubscribe -np 1 ./mpi_mat_mul 4000 400 400 4000
mpirun --allow-run-as-root --oversubscribe -np 2 ./mpi_mat_mul 4000 400 400 4000
mpirun --allow-run-as-root --oversubscribe -np 4 ./mpi_mat_mul 4000 400 400 4000
```

---

## 4. CUDA Matrix Multiplication

### Edit/View Code

```bash
vi cuda_mat_mul.cu
```

### Compile

```bash
nvcc -O3 -arch=sm_75 cuda_mat_mul.cu -o cuda_mat_mul
```

### Run Examples

**Format:**

```bash
./cuda_mat_mul A_rows A_cols B_rows B_cols blockSize
```

```bash
./cuda_mat_mul 4 3 3 5 16
./cuda_mat_mul 40 4000 4000 40 4
./cuda_mat_mul 40 4000 4000 40 8
./cuda_mat_mul 4000 400 400 4000 1
./cuda_mat_mul 4000 400 400 4000 2
./cuda_mat_mul 4000 400 400 4000 4
./cuda_mat_mul 4000 400 400 4000 8
```

---

## Performance Evaluation

This project supports detailed performance analysis:

* Execution time comparison
* Row-wise checksum verification
* Speedup calculation
  
### Graphs (In the Report)

* Threads vs Execution Time - **OpenMP**
* Processes vs Speedup - **MPI**
* Block Size vs Execution Time - **CUDA**
* Serial vs OpenMP vs MPI vs CUDA - **Overall Comparison**

---

## Requirements

### GCC

```bash
sudo apt install build-essential
```

### OpenMP

Included with GCC.

### MPI (MPICH/OpenMPI)

```bash
sudo apt install mpich
```

### CUDA Toolkit

Install via NVIDIA website (Linux/Windows/WSL2) or run CUDA on **Google Colab**.

---

## Demonstration Video

**YouTube link:** https://youtu.be/odvmAwtQgtg

---

## Note:
Full matrix printing is enabled only when all dimensions are ≤ 10 to avoid oversized console output.
Row-wise sums are printed only when result matrix (C) dimensions are ≤ 50 to keep output readable for large matrices.
Otherwise, only Checksum (sum of all elements) is printed.

---
