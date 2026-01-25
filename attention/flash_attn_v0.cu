#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CEIL(a, b)    (((a) + (b) - 1) / (b))
#define OFFSET(row, col, nrows, ncols)  ((row) * (ncols) + (col))

// Bc: Block size for columns (K, V)
// Br: Block size for rows (Q)
template<const int Bc, const int Br>
__global__ void attn_forward(
    const float* Q, const float* K, const float* V,
    const int B, const int NH, const int N, const int d,
    const float scale,
    float* L, float* M, float* O
) {
    // 1. IDENTIFY WORKLOAD
    // gridDim.x = N / Br (row tiles)
    // gridDim.y = NH (heads)
    // gridDim.z = B  (batches)
    int q_tile_idx = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    int tid = threadIdx.x;

    // 2. CALCULATE OFFSETS
    // Offset for current batch and head in the [B, NH, N, d] tensor
    long long head_offset = (long long)(b * NH + h) * (N * d);
    // Offset for the row-wise statistics [B, NH, N]
    long long stats_offset = (long long)(b * NH + h) * N;

    // Pointers to the specific head's data
    const float* Q_head = Q + head_offset;
    const float* K_head = K + head_offset;
    const float* V_head = V + head_offset;
    float* O_head = O + head_offset;

    // 3. SHARED MEMORY ALLOCATION
    // We allocate enough space for one tile of Q, one of K, and one of V.
    extern __shared__ float sram[];
    float* Qi = sram;                // Size: [Br * d]
    float* Kj = &sram[Br * d];       // Size: [Bc * d]
    float* Vj = &sram[Br * d + Bc * d]; // Size: [Bc * d]

    // 4. LOAD Q-TILE INTO SRAM
    // Each thread block stays on the same Q-tile for its entire life.
    for (int i = tid; i < Br * d; i += blockDim.x) {
        int row = i / d;
        int col = i % d;
        if ((q_tile_idx * Br + row) < N) {
            Qi[i] = Q_head[(q_tile_idx * Br + row) * d + col];
        } else {
            Qi[i] = 0; // Padding for safety
        }
    }
    __syncthreads();

    // 5. INITIALIZE LOCAL STATISTICS
    // Each thread processes specific rows within the Br block.
    // If blockDim.x == Br, each thread handles exactly one row.
    float row_m = -INFINITY; // Local running max
    float row_l = 0.0f;      // Local running sum (denominator)

    // 6. OUTER LOOP: ITERATE OVER K/V TILES
    int num_kv_tiles = (N + Bc - 1) / Bc;
    for (int j = 0; j < num_kv_tiles; j++) {
        
        // Load K and V tiles into SRAM
        for (int i = tid; i < Bc * d; i += blockDim.x) {
            int row = i / d;
            int col = i % d;
            int global_kv_row = j * Bc + row;
            if (global_kv_row < N) {
                Kj[i] = K_head[global_kv_row * d + col];
                Vj[i] = V_head[global_kv_row * d + col];
            } else {
                Kj[i] = 0; Vj[i] = 0;
            }
        }
        __syncthreads();

        // 7. COMPUTE ATTENTION (Inside the block)
        if (tid < Br) {
            int q_row_local = tid;
            float new_m = row_m;

            // Step A: Find max of current block to update row_m
            for (int k = 0; k < Bc; k++) {
                float score = 0;
                for (int dim = 0; dim < d; dim++) {
                    score += Qi[q_row_local * d + dim] * Kj[k * d + dim];
                }
                score *= scale;
                if (score > new_m) new_m = score;
            }

            // Step B: Update row_l and Output O
            float p_scale = expf(row_m - new_m); // Rescale factor for previous values
            float batch_l = 0;

            for (int k = 0; k < Bc; k++) {
                float score = 0;
                for (int dim = 0; dim < d; dim++) {
                    score += Qi[q_row_local * d + dim] * Kj[k * d + dim];
                }
                float p_ij = expf(score * scale - new_m);
                batch_l += p_ij;

                // Accumulate into O using the online update formula
                for (int dim = 0; dim < d; dim++) {
                    // This line rescales the PREVIOUS sum in O and adds the NEW tile's contribution
                    float* o_ptr = &O_head[(q_tile_idx * Br + q_row_local) * d + dim];
                    *o_ptr = (*o_ptr * p_scale) + (p_ij * Vj[k * d + dim]);
                }
            }

            // Step C: Update running statistics for the next K/V tile
            row_l = (row_l * p_scale) + batch_l;
            row_m = new_m;
        }
        __syncthreads(); // Sync before loading next K/V tile
    }

    // 8. FINAL NORMALIZATION
    // After seeing all K/V tiles, divide by the final denominator row_l
    if (tid < Br && (q_tile_idx * Br + tid) < N) {
        int final_row = q_tile_idx * Br + tid;
        for (int dim = 0; dim < d; dim++) {
            O_head[final_row * d + dim] /= row_l;
        }
        // Store stats for backward pass
        L[stats_offset + final_row] = row_l;
        M[stats_offset + final_row] = row_m;
    }
}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    /* Q is sharded into Tr * Br blocks */
    /* KV are sharded into Tc * Bc blocks */
    const int Bc = 32;
    const int Br = 32;

    const int batch_size = Q.size(0);
    const int nheads = Q.size(1);
    const int seq_len = Q.size(2);
    const int d_k = Q.size(3);

    const float softmax_scale = 1.0f / sqrt(d_k);

    auto O = torch::zeros_like(Q);
    auto l = torch::zeros({batch_size, nheads, seq_len});
    auto m = torch::full({batch_size, nheads, seq_len}, -INFINITY);

    torch::Device device(torch::kCUDA);
    l = l.to(device); m = m.to(device);

    // TODO: 
    const int sram_size = (3 * Bc * d_k * sizeof(float)) + (Bc * Br * sizeof(float));
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \\n", max_sram_size, sram_size);

    dim3 grid(seq_len / Br, nheads, batch_size);
    dim3 block_size(128);

    attn_forward<Bc, Br><<<grid_size, block_size, sram_size>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        batch_size, nheads, seq_len, d_k,
        softmax_scale,
        l.data_ptr<float>(),
        m.data_ptr<float>(),
        O.data_ptr<float>()
    );
    return O;
}