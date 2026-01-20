#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <math.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

#define CEIL(a, b) (((a) + (b) - 1) / (b))

__global__ void reduce_v0(float* A, float* B, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0.0f;
    if (idx < size) {
        val = A[idx];
    }
    
    // Using atomicAdd for simplicity in this example
    // Note: A real optimized kernel would use shared memory/warp shuffles first
    if (val != 0.0f) {
        atomicAdd(B, val);
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("usage: ./reduce_v0 [N]\n");
        exit(0);
    }

    size_t N = atoi(argv[1]);
    size_t bytes = N * sizeof(float);

    float* h_A = (float*)malloc(bytes);
    for (size_t i = 0; i < N; i++) h_A[i] = 1.0f; // Each element is 1.0

    float *d_A, *B;
    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&B, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));

    int nIter = 100;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /* 1. Custom Kernel Benchmark */
    float custom_res = 0;
    float msec_custom = 0;
    int BLOCK_SIZE = 256;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < nIter; i++) {
        CHECK_CUDA(cudaMemset(B, 0, sizeof(float)));
        dim3 block_size(BLOCK_SIZE);
        dim3 grid_size(CEIL(N, BLOCK_SIZE));
        reduce_v0<<<grid_size, block_size>>>(d_A, B, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&msec_custom, start, stop));
    CHECK_CUDA(cudaMemcpy(&custom_res, B, sizeof(float), cudaMemcpyDeviceToHost));

    /* 2. Thrust Reduction Benchmark */
    float thrust_res = 0;
    float msec_thrust = 0;
    
    // Wrap raw pointer in thrust device_ptr
    thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(d_A);

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < nIter; i++) {
        // thrust::reduce(input_start, input_end, initial_value, binary_op)
        thrust_res = thrust::reduce(dev_ptr, dev_ptr + N, 0.0f, thrust::plus<float>());
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&msec_thrust, start, stop));

    /* Results */
    printf("Total Elements: %zu\n", N);
    printf("Custom Sum: %f, Time: %.3f ms\n", custom_res, msec_custom / nIter);
    printf("Thrust Sum: %f, Time: %.3f ms\n", thrust_res, msec_thrust / nIter);
    
    printf("ratio = %f\n", msec_thrust / msec_custom);

    cudaFree(d_A);
    cudaFree(B);
    free(h_A);
    return 0;
}