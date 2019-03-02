#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

int A[1024][1024];
int B[1024][1024];
int C[1024][1024];

int main(void)
{
	srand(time(NULL));

	for(int i = 0; i < 1024; i++) {
		for(int j = 0; j < 1024; j++) {
			A[i][j] = rand() % 128;
			B[i][j] = rand() % 128;
		}
	}

	for(int i = 0; i < 1024; i++) {
		for(int j = 0; j < 1024; j++) {
			for(int k = 0; k < 1024; k++)
				C[i][j] += A[i][k] * B[k][j];
//			cout << C[i][j] << " ";
		}
//		cout << endl;
	}

	return 0;
}

