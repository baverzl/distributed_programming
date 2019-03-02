/*
 * Calculating prefix sums in serial manner
 *
 * Time complexity: O(n)
 * 
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

typedef enum {
    BROADCAST_TAG,
    MPI_SCAN_TAG
} MESSAGE_TAGS;

int *create_rand_nums(int size)
{
    int i;
    int *res = malloc(sizeof(int) * size);
    assert(res >= 0);
    
    srand(time(NULL));

    for(i = 0; i < size; i++)
        res[i] = rand() % 10;

    return res;
}

int main(int argc, char **argv)
{
    int n;
    int rank, size;

    MPI_Init( &argc, &argv );
    
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    MPI_Barrier( MPI_COMM_WORLD );
    double local_start, local_finish, local_elapsed, elapsed;

    local_start = MPI_Wtime();

    srand(time(NULL));

    if(rank == 0) {
        int sendto;

        printf("rank[%d]: Insert a number of random numbers for prefix sums: ", rank);

        int *rand_num_arrays = create_rand_nums(size);
        for(sendto = 1; sendto < size; sendto++) {
            // blocking send
            MPI_Send(&rand_num_arrays[sendto], 1, MPI_INT, sendto, BROADCAST_TAG, MPI_COMM_WORLD);
        }

        printf("rank[%d] says, \"prefix sum = %d\"\n", rank, rand_num_arrays[0]);
        MPI_Send(&rand_num_arrays[rank], 1, MPI_INT, rank+1, MPI_SCAN_TAG, MPI_COMM_WORLD); // pass it to rank 1
    } else {
        int in, inout;

        // blocking recv
        MPI_Recv(&inout, 1, MPI_INT, 0, BROADCAST_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&in, 1, MPI_INT, rank - 1, MPI_SCAN_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // calculation for prefix sum
        inout += in;

        printf("rank[%d] says, \"prefix sum = %d\"\n", rank, inout);
        if(rank + 1 < size)
            MPI_Send(&inout, 1, MPI_INT, rank + 1, MPI_SCAN_TAG, MPI_COMM_WORLD);
    }

    local_finish = MPI_Wtime();
    local_elapsed = local_finish - local_start;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if(rank == 0)
        printf("Elapsed time = %e seconds\n", elapsed);

    MPI_Finalize();

    return 0;
}
