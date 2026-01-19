#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call) do { \
	cudaError_t e = (call); \
	if (e != cudaSuccess) { \
		fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
		exit(1); \
	} \
} while(0)
#define CEIL(a, b)	(((a) + (b) - 1) / (b))
#define OFFSET(r,c,nrows,ncols) ((r)*(ncols)+(c))

template <const int TILE_SIZE>
__global__ void sgemm_v1(float * __restrict__ A,
                        float * __restrict__ B,
                        float * __restrict__ C,
                        const int M,
                        const int K,
                        const int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int block_row = blockIdx.y * TILE_SIZE;
    int block_col = blockIdx.x * TILE_SIZE;
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    auto load_as = [&](int ph) {
        int a_row = block_row + thread_row;
        int a_col = ph * TILE_SIZE + thread_col;
        if (a_row < M && a_col < K) {
            As[thread_row][thread_col] = A[OFFSET(a_row, a_col, M, K)];
        } else {
            As[thread_row][thread_col] = 0.0f;
        }
    };

    auto load_bs = [&](int ph) {
        int b_row = ph * TILE_SIZE + thread_row;
        int b_col = block_col + thread_col;
        if (b_row < K && b_col < N) {
            Bs[thread_row][thread_col] = B[OFFSET(b_row, b_col, K, N)];
        } else {
            Bs[thread_row][thread_col] = 0.0f;
        }
    };

    auto store_c = [&](float sum) {
        int c_row = block_row + thread_row;
        int c_col = block_col + thread_col;
        if (c_row < M && c_col < N) {
            C[OFFSET(c_row, c_col, M, N)] = sum;
        }
    };

    float sum = 0.0f;
    for (int ph = 0; ph < CEIL(K, TILE_SIZE); ph++) {
        load_as(ph);
        load_bs(ph);
        __syncthreads();
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[thread_row][k] * Bs[k][thread_col];
        }
        __syncthreads();
    }

    store_c(sum);
}

static void init_matrix(float *arr, int rows, int cols) {
    for (size_t i = 0; i < (size_t)rows * (size_t)cols; i++){
        arr[i] = rand() / (float)1147654321;
    }
}

int main(int argc, char** argv) {
    if (argc != 4) {
        printf("usage: ./sgemm_v1 [M] [K] [N]\n");
        exit(0);
    }
    size_t M = atoi(argv[1]);
    size_t K = atoi(argv[2]);
    size_t N = atoi(argv[3]);

    size_t bytes_A = sizeof(float) * M * K;
    size_t bytes_B = sizeof(float) * K * N;
    size_t bytes_C = sizeof(float) * M * N;

    /* Host side malloc and init */
    float* h_A = (float*)malloc(bytes_A);
    float* h_B = (float*)malloc(bytes_B);
    float* h_C = (float*)malloc(bytes_C);
    float* h_C1 = (float*)malloc(bytes_C);
    assert(h_A && h_B && h_C && h_C1);

    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    /* Device side malloc and memcpy*/
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    CHECK_CUDA(cudaMalloc(&d_A, bytes_A));
    CHECK_CUDA(cudaMalloc(&d_B, bytes_B));
    CHECK_CUDA(cudaMalloc(&d_C, bytes_C));
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    /* Test stats, metrics */
    double msecPerMatrixMul[2] = {0, 0};
    double gigaFlops[2] = {0, 0};
    double flopsPerMatrixMul = 2.0 * M * N * K;
    float msecTotal = 0;
    int nIter = 1000;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

	const int TILE_SIZE = 32;
    CHECK_CUDA(cudaEventRecord(start));
    for (int run = 0 ; run < nIter; run ++ ) {
        dim3 block_size(TILE_SIZE, TILE_SIZE);
        dim3 grid_size(CEIL(N, TILE_SIZE), CEIL(M, TILE_SIZE));
        sgemm_v1<TILE_SIZE><<<grid_size, block_size>>>(d_A, d_B, d_C, M, K, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&msecTotal, start, stop));

    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    msecPerMatrixMul[0] = msecTotal / nIter;
    gigaFlops[0] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[0] / 1000.0f);
    printf( "My gemm Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        gigaFlops[0],
        msecPerMatrixMul[0],
        flopsPerMatrixMul);

    /* cuBLAS */
    cublasHandle_t blas_handle;  
    cublasCreate(&blas_handle);
    float alpha = 1.0;
    float beta = 0;

    memset(h_C1, 0, bytes_C);
    CHECK_CUDA(cudaMemcpy(d_C, h_C1, bytes_C, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(start));
    for (int run = 0 ; run < nIter; run ++ ) {
        cublasSgemm(
            blas_handle, 
            CUBLAS_OP_N, 
            CUBLAS_OP_N,
            // M, N, K:
            N, M, K,        // <== Dimensions: Compute C^T (N x M)
            &alpha,
            // Matrix A, lda:
            d_B, N,         // <== The B matrix, transposed (lda = N)
            // Matrix B, ldb:
            d_A, K,         // <== The A matrix, transposed (ldb = K)
            &beta, 
            // Matrix C, ldc:
            d_C, N          // <== Result is C^T (ldc = N)
        );
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&msecTotal, start, stop));

    CHECK_CUDA(cudaMemcpy(h_C1, d_C, bytes_C, cudaMemcpyDeviceToHost));

    msecPerMatrixMul[1] = msecTotal / nIter;
    gigaFlops[1] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[1] / 1000.0f);
    printf( "CuBlas Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        gigaFlops[1],
        msecPerMatrixMul[1],
        flopsPerMatrixMul);

    cublasDestroy(blas_handle); 
    
    double eps = 1.e-6;  // machine zero
    bool correct = true;
    for (int i = 0; i < M * N; i++) {
        int row = i / N;
        int col = i % N;

        double abs_err = fabs(h_C[i] - h_C1[i]);
        double dot_length = M;
        double abs_val = fabs(h_C[i]);
        double rel_err = abs_err / abs_val / dot_length;
        if (rel_err > eps) {
            printf("Error! Custom kernel res[%d][%d]=%.8f, cuBLAS res[%d][%d]=%.8f error term is > %E\n",
                    row, col, h_C[i], col, row, h_C1[i], eps);
            correct = false;
            break;
        }
    }

    printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");
    printf("ratio= %f\n", gigaFlops[0] / gigaFlops[1]);
    // Free Memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C1);
}
