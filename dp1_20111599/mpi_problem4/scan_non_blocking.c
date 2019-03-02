/*
 * Calculating prefix sums in serial manner (non-blocking)
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
int get_rand_number()
{
    static int isFirstCall = 0;
    if(isFirstCall == 0) {
        srand(time(NULL));
        isFirstCall = 1;
    }

    return rand() % 1000;
}
int isMaster(int rank)
{
    return (rank == 0);
}

int main(int argc, char **argv)
{
    int n, i;
    int rank, pool_size;

    MPI_Init( &argc, &argv );
    
    MPI_Comm_size( MPI_COMM_WORLD, &pool_size );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    // The non-blocking call needs one more parameter: MPI_Request *request
    // This is a so called opaque object, which identifies communication operations and 
    // matches (the operation that initiates the communication) with (the operation that terminates it). 
    MPI_Status  *status = malloc(sizeof(MPI_Status) * pool_size);

    MPI_Barrier( MPI_COMM_WORLD );
   
    double local_start, local_finish, local_elapsed, elapsed;

    local_start = MPI_Wtime();


    if(isMaster(rank)) { // master
        int sendto;
        // how many different requests are you going to make?
        MPI_Request *req_handles = malloc(sizeof(MPI_Request) * (pool_size));
        MPI_Status  *status      = malloc(sizeof(MPI_Status) * (pool_size));
        
        int *rand_num = create_rand_nums(pool_size);
        for(sendto = 1; sendto < pool_size; sendto++) {
            // non-blocking send
            MPI_Isend(&rand_num[sendto], 1, MPI_INT, sendto, BROADCAST_TAG, MPI_COMM_WORLD, &req_handles[sendto-1]);
        }
        printf("rank[%d] says, \"prefix sum = %d\"\n", rank, rand_num[0]);
        MPI_Isend(&rand_num[rank], 1, MPI_INT, rank + 1, MPI_SCAN_TAG, MPI_COMM_WORLD, &req_handles[pool_size-1]);

        MPI_Waitall(pool_size, req_handles, status);

        free(req_handles);
        free(status);
    } 
    else { // other nodes
        MPI_Request req_handles[2];
        MPI_Status  status[2];
        int inout, in;

        // Request[0]: non-blocking recv from rank 0
        MPI_Irecv(&inout, 1, MPI_INT, 0, BROADCAST_TAG, MPI_COMM_WORLD, &req_handles[0]);

        // Request[1]: non-blocking recv from rank - 1
        MPI_Irecv(&in, 1, MPI_INT, rank - 1, MPI_SCAN_TAG, MPI_COMM_WORLD, &req_handles[1]);

        // wait because you need in and inout for future calculation
        MPI_Waitall(2, req_handles, status);

        int num_of_received;
        // Status[1] = Outcome of Request[1]
        MPI_Get_count(&status[1], MPI_INT, &num_of_received);
        if(num_of_received == 1)
            inout += in;
        else {
            assert(0);
        }

        printf("rank[%d] says, \"prefix sum = %d\"\n", rank, inout);
        if(rank + 1 < pool_size) {
            // non-blocking send to ( rank + 1 )
            MPI_Isend(&inout, 1, MPI_INT, rank + 1, MPI_SCAN_TAG, MPI_COMM_WORLD, &req_handles[0]);

            MPI_Wait(&req_handles[0], &status[0]);
        }
    }

    local_finish = MPI_Wtime();
    local_elapsed = local_finish - local_start;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if(rank == 0)
        printf("Elapsed time = %e seconds\n", elapsed);

    MPI_Finalize();

    return 0;
}
