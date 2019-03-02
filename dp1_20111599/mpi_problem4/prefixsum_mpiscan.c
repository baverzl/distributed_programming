#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

// Simple MPI_SUM

int main(int argc, char **argv)
{
    int n, i;
    int rank, size;
    int random_int;

    MPI_Comm comm = MPI_COMM_WORLD;

    MPI_Init( &argc, &argv );

    MPI_Comm_rank( comm, &rank );
    MPI_Comm_size( comm, &size );

    MPI_Barrier( comm );
    double local_start, local_finish, local_elapsed, elapsed;

    local_start = MPI_Wtime();

    srand(time(NULL));

    // dependent on the number of processors
    int *a = calloc(size, sizeof(int));

    for(i = 0; i < size; i++)
        a[i] = 1;

    int prefix_sum;
    MPI_Scan( &a[rank], &prefix_sum, 1, MPI_INT, MPI_SUM, comm );

    printf("rank[%d] says, prefix sum is %d\n", rank, prefix_sum);

    local_finish  = MPI_Wtime();
    local_elapsed = local_finish - local_start;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm );

    if(rank == 0)
        printf("Elapsed time = %e seconds\n", elapsed);

    MPI_Finalize();

    return 0;
}
