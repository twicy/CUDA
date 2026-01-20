#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

#define CEIL(a, b) (((a) + (b) - 1) / (b))
#define OFFSET(row, col, nrows, ncols)  ((row) * (ncols) + (col))

// Dummy init function
void init_matrix(float* data, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        data[i] = (float)rand() / (float)RAND_MAX;
    }
}

template <const int TILE_SIZE>
__global__ void transpose_v3(float* A, float* B, int M, int N) {
    int block_row = blockIdx.y * TILE_SIZE;
    int block_col = blockIdx.x * TILE_SIZE;
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;
    int a_row = block_row + thread_row;
    int a_col = block_col + thread_col;

    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];

    if (a_row < M && a_col < N) {
        As[thread_row][thread_col] = A[OFFSET(a_row, a_col, M, N)];
    } else {
        As[thread_row][thread_col] = 0.0f;
    }

    __syncthreads();

    int b_col = block_row + thread_col;
    int b_row = block_col + thread_row;

    if (b_row < N && b_col < M) {
        B[OFFSET(b_row, b_col, N, M)] = As[thread_col][thread_row];
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("usage: ./transpose [M] [N]\n");
        exit(0);
    }

    size_t M = atoi(argv[1]);
    size_t N = atoi(argv[2]);

    size_t bytes = sizeof(float) * M * N;

    /* Host side malloc */
    float* h_A  = (float*)malloc(bytes);
    float* h_B  = (float*)malloc(bytes); // Result from custom kernel
    float* h_B1 = (float*)malloc(bytes); // Result from cuBLAS
    assert(h_A && h_B && h_B1);

    init_matrix(h_A, M, N);

    /* Device side malloc and memcpy */
    float *d_A = NULL, *d_B = NULL;
    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&d_B, bytes));
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));

    /* Timing setup */
    float msecTotal = 0;
    int nIter = 100;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    /* 1. Custom Kernel Execution */
    const int BLOCK_SIZE = 32;
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));

    CHECK_CUDA(cudaEventRecord(start));
    for (int run = 0; run < nIter; run++) {
        transpose_v3<BLOCK_SIZE><<<grid_size, block_size>>>(d_B, d_A, M, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&msecTotal, start, stop));

    CHECK_CUDA(cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost));
    float avg_msec_custom = msecTotal / nIter;

    /* 2. cuBLAS SGEAM Execution */
    cublasHandle_t blas_handle;
    cublasCreate(&blas_handle);
    float alpha = 1.0f;
    float beta = 0.0f;

    // Reset device output
    CHECK_CUDA(cudaMemset(d_B, 0, bytes));

    CHECK_CUDA(cudaEventRecord(start));
    for (int run = 0; run < nIter; run++) {
        // To transpose a Row-Major M x N matrix:
        // cuBLAS sees this as a Column-Major N x M matrix.
        // We ask for a Transpose of that Col-Major N x M matrix.
        // The result is a Col-Major M x N matrix, which is Row-Major N x M.
        cublasSgeam(
            blas_handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            M, N,            // rows, cols of C (column-major)
            &alpha,
            d_A, N,          // lda = number of rows of A_col = N
            &beta,
            d_A, M,
            d_B, M           // ldc = number of rows of C_col = M
        );
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&msecTotal, start, stop));

    CHECK_CUDA(cudaMemcpy(h_B1, d_B, bytes, cudaMemcpyDeviceToHost));
    float avg_msec_cublas = msecTotal / nIter;

    /* Verification & Stats */
    bool correct = true;
    for (int i = 0; i < M * N; i++) {
        if (fabs(h_B[i] - h_B1[i]) > 1e-5) {
            correct = false;
            break;
        }
    }

    printf("Matrix Size: %zu x %zu\n", M, N);
    printf("Custom Kernel:  Time = %.3f msec\n", avg_msec_custom);
    printf("cuBLAS SGEAM:   Time = %.3f msec\n", avg_msec_cublas);
    printf("ratio = %f\n", avg_msec_cublas / avg_msec_custom);
    printf("Result: %s\n", correct ? "PASS" : "FAIL");

    // Cleanup
    cublasDestroy(blas_handle);
    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);
    free(h_B1);

    return 0;
}