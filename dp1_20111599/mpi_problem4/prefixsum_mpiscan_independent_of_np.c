#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void addem(int *invec, int *inoutvec, int *len, MPI_Datatype *dtype)
{
    int i;
    inoutvec[0] = invec[*len - 1];
    for(i = 1; i < *len; i++)
        inoutvec[i] += inoutvec[i-1];
}

int main(int argc, char **argv)
{
    int n, i;
    int rank, size;
    int random_int;

    MPI_Op op_addem;
    MPI_Comm comm = MPI_COMM_WORLD;

    MPI_Init( &argc, &argv );

    MPI_Op_create( (MPI_User_function *) addem, 0, &op_addem );
    
    MPI_Comm_rank( comm, &rank );
    MPI_Comm_size( comm, &size );

    MPI_Barrier( comm );
    double local_start, local_finish, local_elapsed, elapsed;

    local_start = MPI_Wtime();

    srand(time(NULL));

    n = size * 100;

    int *a = calloc(n, sizeof(int));

    for(i = 0; i < n; i++)
        a[i] = 1;

    if(n % size != 0) {
        printf("MPI_Scan cannot process it\n");
        return -1;
    }

    int block_size = n / size;
    int from = rank * block_size;

    if(rank == 0) {
        for(i = 1; i  < block_size; i++)
            a[i] += a[i-1];
    }

    int *prefix_sum = calloc(block_size, sizeof(int));
    MPI_Scan( &a[ from ], prefix_sum, block_size, MPI_INT, op_addem, comm );

    printf("rank[%d] says,\n", rank);
    for(i = 0; i < block_size; i++)
        printf("%d ", prefix_sum[i]);
    printf("\n");

    local_finish  = MPI_Wtime();
    local_elapsed = local_finish - local_start;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm );

    if(rank == 0)
        printf("Elapsed time = %e seconds\n", elapsed);

    MPI_Finalize();

    return 0;
}
