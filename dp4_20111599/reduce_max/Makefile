reduce_max:
	gcc reduce_max_cpu.c -o reduce_max_cpu -g
	nvcc reduce_max.cu -o reduce_max
	nvcc reduce_max_non_divergent.cu -o reduce_max_non_divergent
	nvcc reduce_max_sequential_addressing.cu -o reduce_max_sequential_addressing
clean:
	rm reduce_max_cpu
	rm reduce_max
	rm reduce_max_non_divergent
	rm reduce_max_sequential_addressing
