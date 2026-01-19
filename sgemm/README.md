# SGEMM Kernels

## Test Result ($M = N = K = 4096$)

|Program|Perf (GFlop/s)|\% cuBLAS|Desc|
|-|-|-|-|
|sgemm_v0|5331.63|16.53|Basic Inner Product|
|sgemm_v1|7336.45|23.37|Shared Memory Tiling|
|sgemm_v2|22684.39|69.00|Thread Tiling, Outer Product|
|sgemm_v3|24677.72|75.99|Reduced bank conflict|
|sgemm_v4|27239.50|85.00|Vectorized Load|
|sgemm_v5|27049.20|82.61|Double Buffering|
|sgemm_v6|27339.51|86.32|Warp Tiling|

## Future Work

- [ ] Revise Double Buffering
- [ ] Bank Conflict of Registers
- [ ] Double Buffering for Registers
- [ ] PTX Related
- [ ] Tensor Core
