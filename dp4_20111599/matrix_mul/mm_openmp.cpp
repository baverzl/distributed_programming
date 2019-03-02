#include <iostream>
#include <cstdlib>
#include <ctime>
#include <omp.h>

using namespace std;

int A[1024][1024];
int B[1024][1024];
int C[1024][1024];

int main(void)
{
	int i, j, k;

	srand(time(NULL));

	for(int i = 0; i < 1024; i++) {
		for(int j = 0; j < 1024; j++) {
			A[i][j] = rand() % 128;
			B[i][j] = rand() % 128;
		}
	}

	cout << "Max number of threads: " << omp_get_max_threads() << endl;

	#pragma omp parallel
	cout << "Number of threads: " << omp_get_num_threads() << endl;

	double wtime;
	wtime = omp_get_wtime();

	#pragma omp parallel for private(k,j)
	for(i = 0; i < 1024; i++) {
		for(j = 0; j < 1024; j++) {
			for(k = 0; k < 1024; k++)
				C[i][j] += A[i][k] * B[k][j];
//			cout << C[i][j] << " ";
		}
//		cout << endl;
	}

	wtime = omp_get_wtime() - wtime;

	cout << "Execution time: " << wtime << endl;

	return 0;
}
