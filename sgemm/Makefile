all:
	nvcc sgemm_v0.cu -g -lcublas -o sgemm_v0
	nvcc sgemm_v1.cu -g -lcublas -o sgemm_v1
	nvcc sgemm_v2.cu -g -lcublas -o sgemm_v2
	nvcc sgemm_v3.cu -g -lcublas -o sgemm_v3
	nvcc sgemm_v4.cu -g -lcublas -o sgemm_v4
	nvcc sgemm_v5.cu -g -lcublas -o sgemm_v5
	nvcc sgemm_v6.cu -g -lcublas -o sgemm_v6
clean:
	rm -rf *.o sgemm_v0 sgemm_v1 sgemm_v2 sgemm_v3 sgemm_v4 sgemm_v5 sgemm_v6
