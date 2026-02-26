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
#define OFFSET(r,c,ncols) ((r)*(ncols)+(c))
#define FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

__global__ __launch_bounds__(256, 2)
void sgemm_128x128x8_kernel_v1(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N)
{
    __shared__ __align__(16) float AsT[8][132];
    __shared__ __align__(16) float Bs[8][128];
    __shared__ __align__(16) float Cs[16][64];

    __align__(16) float A_frag[8];
    __align__(16) float B_frag[8];
    __align__(16) float acc[8][8] = {0.0f};

    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;
    const int block_row = 128 * blockIdx.y;
    const int block_col = 128 * blockIdx.x;
    /* Loading global memory to reg */
    float4 A_ldg_reg;
    float4 B_ldg_reg;
    const int A_thread_row = (threadIdx.x / 8) * 4;
    const int A_thread_col = threadIdx.x % 8;
    const int A_row = block_row + A_thread_row;
    /* First element to be loaded */
    const int B_thread_row = threadIdx.x / 32;
    const int B_thread_col = threadIdx.x % 32;
    const int B_col = block_col + B_thread_col;
    /* Computational */
    const int warp_row = (warpId / 2) * 4 * 8;
    const int warp_col = (warpId % 2) * 4 * 16;
    const int frag_row = (laneId / 16) * 2  + (laneId % 2);
    const int frag_col = (laneId / 2) % 8;
    const int A_frag_row = warp_row + frag_row * 4;
    const int B_frag_col = warp_col + frag_col * 4;

    auto ld_g2r_a = [&] (int ph) {
        int col = ph * 8 + A_thread_col;
        A_ldg_reg.x = (A_row < M && col < K) ? A[OFFSET(A_row, col, K)] : 0.0f;
        A_ldg_reg.y = (A_row + 1 < M && col < K) ? A[OFFSET(A_row + 1, col, K)] : 0.0f;
        A_ldg_reg.z = (A_row + 2 < M && col < K) ? A[OFFSET(A_row + 2, col, K)] : 0.0f;
        A_ldg_reg.w = (A_row + 3 < M && col < K) ? A[OFFSET(A_row + 3, col, K)] : 0.0f;
    };

    auto ld_g2r_b = [&] (int ph) {
        int row = ph * 8 + B_thread_row;
        B_ldg_reg.x = (row < K && B_col < N) ? B[OFFSET(row, B_col, N)] : 0.0f;
        B_ldg_reg.y = (row < K && B_col + 32 < N) ? B[OFFSET(row, B_col + 32, N)] : 0.0f;
        B_ldg_reg.z = (row < K && B_col + 64 < N) ? B[OFFSET(row, B_col + 64, N)] : 0.0f;
        B_ldg_reg.w = (row < K && B_col + 96 < N) ? B[OFFSET(row, B_col + 96, N)] : 0.0f;
    };

    auto st_r2s_a = [&] () {
        FLOAT4(AsT[A_thread_col][A_thread_row]) = A_ldg_reg;
    };

    auto st_r2s_b = [&] () {
        Bs[B_thread_row][B_thread_col] = B_ldg_reg.x;
        Bs[B_thread_row][B_thread_col + 32] = B_ldg_reg.y;
        Bs[B_thread_row][B_thread_col + 64] = B_ldg_reg.z;
        Bs[B_thread_row][B_thread_col + 96] = B_ldg_reg.w;
    };

    auto mma = [&] () {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                acc[i][j] += A_frag[i] * B_frag[j];
            }
        }
    };

    auto ld_frag_a = [&] (int k) {
        FLOAT4(A_frag[0]) = FLOAT4(AsT[k][A_frag_row]);
        FLOAT4(A_frag[4]) = FLOAT4(AsT[k][A_frag_row + 16]);
    };

    auto ld_frag_b = [&] (int k) {
        FLOAT4(B_frag[0]) = FLOAT4(Bs[k][B_frag_col]);
        FLOAT4(B_frag[4]) = FLOAT4(Bs[k][B_frag_col + 32]);
    };

    int cs_warp_row = (warpId / 2) * 4;
    int cs_warp_col = (warpId % 2) * 32;
    /* Each thread takes care of 4 thread blocks */
    auto st_r2s_c = [&] (int i, int j, int row) {
        FLOAT4(Cs[cs_warp_row + frag_row][cs_warp_col + frag_col * 4]) = FLOAT4(acc[4 * i + row][4 * j]);
    };

    int reorg_cs_warp_row = (warpId / 2) * 4;
    int reorg_cs_warp_col = (warpId % 2) * 32;
    int reorg_frag_row = laneId / 8;
    int reorg_frag_col = laneId % 8;
    int cs_row = reorg_cs_warp_row + reorg_frag_row;
    int cs_col = reorg_cs_warp_col + reorg_frag_col * 4;
    auto st_s2g_c = [&] (int i, int j, int row) {
        int c_row = block_row + warp_row + i * 16 + reorg_frag_row * 4 + row;
        int c_col = block_col + warp_col + j * 32 + reorg_frag_col * 4;
        if (c_row < M && c_col < N) {
            FLOAT4(C[OFFSET(c_row, c_col, N)]) = FLOAT4(Cs[cs_row][cs_col]);
        }
    };

    #pragma unroll
    for (int ph = 0; ph < CEIL(K, 8); ph++) {
        ld_g2r_a(ph);
        ld_g2r_b(ph);
        st_r2s_a();
        st_r2s_b();

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < 8; k++) {
            ld_frag_a(k);
            ld_frag_b(k);
            mma();
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            #pragma unroll
            for (int row = 0; row < 4; row++) {
                st_r2s_c(i, j, row);
                __syncwarp();
                st_s2g_c(i, j, row);
                __syncwarp();
            }
        }
    }
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

    CHECK_CUDA(cudaEventRecord(start));
    for (int run = 0 ; run < nIter; run ++ ) {
        dim3 block_size(256);
        dim3 grid_size(CEIL(N, 128), CEIL(M, 128));
        if (M % 4 == 0 && N % 4 == 0 && K % 4 == 0) {
            sgemm_128x128x8_kernel_v1<<<grid_size, block_size>>>(d_A, d_B, d_C, M, K, N);
        } else {
            sgemm_128x128x8_kernel_v1<<<grid_size, block_size>>>(d_A, d_B, d_C, M, K, N);
        }
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
