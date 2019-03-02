scan:
	mpicc scan_non_blocking.c -o scan_non_blocking
	mpicc scan_blocking.c -o scan_blocking
	mpicc prefixsum_mpiscan.c -o prefixsum_mpiscan
	mpicc prefixsum_mpiscan_independent_of_np.c -o prefixsum_mpiscan_independent_of_np
clean:
	rm prefixsum_mpiscan scan_blocking scan_non_blocking prefixsum_mpiscan_independent_of_np
