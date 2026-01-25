#include <cfloat>
struct __align__(8) MD {
    float m;
    float d;
};

__device__ MD reduce_md_op(MD a, MD b) {
    MD ret;
    ret.m = fmax(a.m, b.m);
    ret.d = a.d * __expf(ret.m - a.m) + b.d * __expf(ret.m - b.m);
    return ret;
}

__device__ MD warp_reduce_md(MD &md) {
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        MD other_md = __shfl_down_sync(mask, md, offset);
        md = reduce_md_op(md, other_md);
    }
    return md;
}

/* One block handles everything */
__global__ void softmax(float *A, float *B, int N) {
    __shared__ MD b_md;
    int tid = threadIdx.x;
    MD t_md;
    t_md.m = FLT_MIN;
    t_md.d = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        MD tmp_md;
        tmp_md.m = A[i];
        tmp_md.d = 1.0f;
        t_md = reduce_md_op(t_md, tmp_md);
    }

    // TODO: block_reduce

    if (tid == 0) {
        b_md = t_md;
    }

    for (int i = tid; i < N; i += blockDim.x) {
        B[i] = __expf(A[i] - b_md.m) / b_md.d;
    }

}