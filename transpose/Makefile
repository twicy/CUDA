all:
	nvcc transpose_v0.cu -g -lcublas -o transpose_v0
	nvcc transpose_v1.cu -g -lcublas -o transpose_v1
	nvcc transpose_v2.cu -g -lcublas -o transpose_v2
	nvcc transpose_v3.cu -g -lcublas -o transpose_v3
	nvcc transpose_v4.cu -g -lcublas -o transpose_v4
clean:
	rm -rf *.o transpose_v0 transpose_v1 transpose_v2 transpose_v3 transpose_v4