# Transpose Kernels

## Test Result ($M = N = 4096$)

|Program|Time (msec)|\% cuBLAS|Desc|
|-|-|-|-|
|sgemm_v0|0.377|39.21|A coalesced read|
|sgemm_v1|0.202|73.13|B coalesced write|
|sgemm_v2|0.170|86.56|A, B both coalesced with shared memory|
|sgemm_v3|0.154|96.14|Padding to reduce bank conflict|
|sgemm_v4|0.158|93.64|Swizzling|

## Future Work

- [ ] Vectorized ld/st
- [ ] Each thread transpose multiple elements
- [ ] The math part, why Swizzling works?