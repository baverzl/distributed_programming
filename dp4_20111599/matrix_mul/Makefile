mm:
	nvcc mm.cu -o mm
	g++ mm_naive.cpp -o mm_naive
	g++ mm_openmp.cpp -o mm_openmp -fopenmp
clean:
	rm mm_naive
	rm mm_openmp
	rm mm
