/* reduction sum sequential version */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

const int N = 1e5;

int reduce_max(int *data) {

	// do reduction in shared mem
	for(unsigned int s = 1; s < N; s *= 2) {
		for(int i = 0; i + s < N; i += 2 * s)
			data[i] = (data[i] < data[i + s])? data[i + s] : data[i];
	}
	return data[0];
}

int main(void)
{
	int data[N];

	srand(time(NULL));

	for(int i = 0; i < N; i++)
		data[i] = rand() % 128;		

	printf("Max: %d\n", reduce_max(data) ) ;

	return 0;

}

