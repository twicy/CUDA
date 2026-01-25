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
#define FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

template <const int BM, const int BK, const int BN,
        const int WM, const int WN,
        const int TM, const int TN>
__global__ void sgemm_v6_nonvec(float * __restrict__ A,
                        float * __restrict__ B,
                        float * __restrict__ C,
                        const int M,
                        const int K,
                        const int N) {
    __shared__ float AsT[2][BK][BM];
    __shared__ float Bs[2][BK][BN];

    float Ar[TM] = {0.0f};
    float Br[TN] = {0.0f};
    float acc[TM][TN] = {{0.0f}};

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int warpId = tid / warpSize;
    int laneId = tid % warpSize;
    int nwarps_per_block_row = BN / WN;
    int warpy = warpId / nwarps_per_block_row;
    int warpx = warpId % nwarps_per_block_row;
    int warp_row = warpy * WM;
    int warp_col = warpx * WN;
    int nthreads_per_warp_row = WN / TN;
    int nthreads_per_warp_col = WM / TM;
    int thready = laneId / nthreads_per_warp_row;
    int threadx = laneId % nthreads_per_warp_row;

    int nthreads = blockDim.x * blockDim.y;

    auto load_ast = [&](int ph, int stage) {
        int nelements = BM * BK;
        for (int i = tid; i < nelements; i += nthreads) {
            int shmem_row = i / BK;
            int shmem_col = i % BK;
            int a_row = block_row + shmem_row;
            int a_col = ph * BK + shmem_col;
            if (a_row < M && a_col < K) {
                AsT[stage][shmem_col][shmem_row] = A[OFFSET(a_row, a_col, M, K)];
            } else {
                AsT[stage][shmem_col][shmem_row] = 0.0f;
            }
        }
    };

    auto load_bs = [&](int ph, int stage) {
        int nelements = BK * BN;
        for (int i = tid; i < nelements; i += nthreads) {
            int shmem_row = i / BN;
            int shmem_col = i % BN;
            int b_row = ph * BK + shmem_row;
            int b_col = block_col + shmem_col;
            if (b_row < K && b_col < N) {
                Bs[stage][shmem_row][shmem_col] = B[OFFSET(b_row, b_col, K, N)];
            } else {
                Bs[stage][shmem_row][shmem_col] = 0.0f;
            }
        }
    };

    auto load_ar = [&] (int k, int stage) {
        for (int reg_base = 0, piece_base = thready * 4; reg_base < TM; reg_base += 4, piece_base += nthreads_per_warp_col * 4) {
            FLOAT4(Ar[reg_base]) = FLOAT4(AsT[stage][k][warp_row + piece_base]);
        }
    };

    auto load_br = [&] (int k, int stage) {
        for (int reg_base = 0, piece_base = threadx * 4; reg_base < TN; reg_base += 4, piece_base += nthreads_per_warp_row * 4) {
            FLOAT4(Br[reg_base]) = FLOAT4(Bs[stage][k][warp_col + piece_base]);
        }
    };

    auto mma = [&] () {
        #pragma unroll
        for (int m = 0; m < TM; m++) {
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                acc[m][n] += Ar[m] * Br[n];
            }
        }
    };

    auto store_c = [&] () {
        int row_piece_base = thready * 4;
        int col_piece_base = threadx * 4;
        #pragma unroll
        for (int m = 0; m < TM; m++) {
            int m_piece_block = m / 4;
            int m_piece_offset = m % 4;
            int global_row = block_row + warp_row + row_piece_base +
                m_piece_block * (nthreads_per_warp_col * 4) + m_piece_offset;
            if (global_row >= M) break;
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                int n_piece_block = n / 4;
                int n_piece_offset = n % 4;
                int global_col = block_col + warp_col + col_piece_base +
                    n_piece_block * (nthreads_per_warp_row * 4) + n_piece_offset;
                if (global_col >= N) break;
                C[OFFSET(global_row, global_col, M, N)] = acc[m][n];
            }
        }
    };

    int ph = 0;
    int curr_stage = 0, other_stage = 1;
    load_ast(ph, curr_stage);
    load_bs(ph, curr_stage);
    __syncthreads();

    for (ph = 1; ph < CEIL(K, BK); ph++) {
        load_ast(ph, other_stage);
        load_bs(ph, other_stage);

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            load_ar(k, curr_stage);
            load_br(k, curr_stage);
            mma();
        }
        __syncthreads();
        curr_stage = other_stage;
        other_stage = 1 - curr_stage;
    }

    #pragma unroll
    for (int k = 0; k < BK; k++) {
        load_ar(k, curr_stage);
        load_br(k, curr_stage);
        mma();
    }

    __syncthreads();
    store_c();
}

template <const int BM, const int BK, const int BN,
        const int WM, const int WN,
        const int TM, const int TN>
__global__ void sgemm_v6_vec(float * __restrict__ A,
                        float * __restrict__ B,
                        float * __restrict__ C,
                        const int M,
                        const int K,
                        const int N) {
    __shared__ float AsT[2][BK][BM];
    __shared__ float Bs[2][BK][BN];

    float Ar[TM] = {0.0f};
    float Br[TN] = {0.0f};
    float acc[TM][TN] = {{0.0f}};

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int warpId = tid / warpSize;
    int laneId = tid % warpSize;
    int nwarps_per_block_row = BN / WN;
    int warpy = warpId / nwarps_per_block_row;
    int warpx = warpId % nwarps_per_block_row;
    int warp_row = warpy * WM;
    int warp_col = warpx * WN;
    int nthreads_per_warp_row = WN / TN;
    int nthreads_per_warp_col = WM / TM;
    int thready = laneId / nthreads_per_warp_row;
    int threadx = laneId % nthreads_per_warp_row;

    int nthreads = blockDim.x * blockDim.y;

    auto load_ast = [&](int ph, int stage) {
        int nelements = BM * BK;
        for (int i = 4 * tid; i < nelements; i += 4 * nthreads) {
            int shmem_row = i / BK;
            int shmem_col = i % BK;
            int a_row = block_row + shmem_row;
            int a_col = ph * BK + shmem_col;
            if (a_row < M && a_col < K) {
                float4 a4 = FLOAT4(A[OFFSET(a_row, a_col, M, K)]);
                AsT[stage][shmem_col][shmem_row] = a4.x;
                AsT[stage][shmem_col + 1][shmem_row] = a4.y;
                AsT[stage][shmem_col + 2][shmem_row] = a4.z;
                AsT[stage][shmem_col + 3][shmem_row] = a4.w;
            } else {
                AsT[stage][shmem_col][shmem_row] = 0.0f;
                AsT[stage][shmem_col + 1][shmem_row] = 0.0f;
                AsT[stage][shmem_col + 2][shmem_row] = 0.0f;
                AsT[stage][shmem_col + 3][shmem_row] = 0.0f;
            }
        }
    };

    auto load_bs = [&](int ph, int stage) {
        int nelements = BK * BN;
        for (int i = 4 * tid; i < nelements; i += 4 * nthreads) {
            int shmem_row = i / BN;
            int shmem_col = i % BN;
            int b_row = ph * BK + shmem_row;
            int b_col = block_col + shmem_col;
            if (b_row < K && b_col < N) {
                FLOAT4(Bs[stage][shmem_row][shmem_col]) = FLOAT4(B[OFFSET(b_row, b_col, K, N)]);
            } else {
                FLOAT4(Bs[stage][shmem_row][shmem_col]) = {0.0f, 0.0f, 0.0f, 0.0f};
            }
        }
    };

    auto load_ar = [&] (int k, int stage) {
        for (int reg_base = 0, piece_base = thready * 4; reg_base < TM; reg_base += 4, piece_base += nthreads_per_warp_col * 4) {
            FLOAT4(Ar[reg_base]) = FLOAT4(AsT[stage][k][warp_row + piece_base]);
        }
    };

    auto load_br = [&] (int k, int stage) {
        for (int reg_base = 0, piece_base = threadx * 4; reg_base < TN; reg_base += 4, piece_base += nthreads_per_warp_row * 4) {
            FLOAT4(Br[reg_base]) = FLOAT4(Bs[stage][k][warp_col + piece_base]);
        }
    };

    auto mma = [&] () {
        #pragma unroll
        for (int m = 0; m < TM; m++) {
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                acc[m][n] += Ar[m] * Br[n];
            }
        }
    };

    auto store_c = [&] () {
        int row_piece_base = thready * 4;
        int col_piece_base = threadx * 4;
        #pragma unroll
        for (int m = 0; m < TM; m++) {
            int m_piece_block = m / 4;
            int m_piece_offset = m % 4;
            int global_row = block_row + warp_row + row_piece_base +
                m_piece_block * (nthreads_per_warp_col * 4) + m_piece_offset;
            if (global_row >= M) break;
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                int n_piece_block = n / 4;
                int n_piece_offset = n % 4;
                int global_col = block_col + warp_col + col_piece_base +
                    n_piece_block * (nthreads_per_warp_row * 4) + n_piece_offset;
                if (global_col >= N) break;
                C[OFFSET(global_row, global_col, M, N)] = acc[m][n];
            }
        }
    };

    int ph = 0;
    int curr_stage = 0, other_stage = 1;
    load_ast(ph, curr_stage);
    load_bs(ph, curr_stage);
    __syncthreads();

    for (ph = 1; ph < CEIL(K, BK); ph++) {
        load_ast(ph, other_stage);
        load_bs(ph, other_stage);

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            load_ar(k, curr_stage);
            load_br(k, curr_stage);
            mma();
        }
        __syncthreads();
        curr_stage = other_stage;
        other_stage = 1 - curr_stage;
    }

    #pragma unroll
    for (int k = 0; k < BK; k++) {
        load_ar(k, curr_stage);
        load_br(k, curr_stage);
        mma();
    }

    __syncthreads();
    store_c();
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

	const int BM = 128, BN = 128, BK = 8;
    const int WM = 32, WN = 64;
    const int TM = 8, TN = 8;
    CHECK_CUDA(cudaEventRecord(start));
    for (int run = 0 ; run < nIter; run ++ ) {
        dim3 block_size(BN / TN, BM / TM);
        dim3 grid_size(CEIL(N, BN), CEIL(M, BM));
        if (M % 4 == 0 && N % 4 == 0 && K % 4 == 0) {
            sgemm_v6_vec<BM, BK, BN, WM, WN, TM, TN><<<grid_size, block_size>>>(d_A, d_B, d_C, M, K, N);
        } else {
            sgemm_v6_nonvec<BM, BK, BN, WM, WN, TM, TN><<<grid_size, block_size>>>(d_A, d_B, d_C, M, K, N);
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
