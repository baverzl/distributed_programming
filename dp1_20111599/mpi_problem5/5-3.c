#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int ELEM_PER_PROC;

int collapse(int n)
{
    int sum = 0;
    while(n) {
        sum += n % 10;
        n /= 10;
    }
    return sum;
}
int ultimate_collapse(int n)
{
    int sum, loop_cnt;
    while(1) {
        // n is a single digit from 0 ~ 9 
        if(n / 10 == 0) {
           sum = n;
           break;
        }
        n = collapse(n);
    }
    return sum;
}
int is_master(int rank)
{
    return (rank == 0);
}

int main(int argc, char *argv[])
{
    int rank, size, i, n;
    int *nums = NULL;
    int comm_size;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    comm_size = size;

    if(is_master(rank))
    {
        scanf("%d", &n);
        nums = calloc( n, sizeof(int) );
        for(i = 0; i < n; i++)
            scanf("%d", &nums[i]);
    }

    MPI_Barrier( MPI_COMM_WORLD );

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(n < size) {
        size = n;
    }

    if(rank != size - 1)
        ELEM_PER_PROC = n / size;
    else
        ELEM_PER_PROC = n / size + n % size;

    int *sub_nums = calloc(ELEM_PER_PROC, sizeof(int));
    assert(sub_nums != NULL);

    if(is_master(rank)) 
    {
        int to;
        MPI_Request *reqs  = calloc(size - 1, sizeof(MPI_Request));
        MPI_Status  *stats = calloc(size - 1, sizeof(MPI_Status));

        for(i = 0; i < ELEM_PER_PROC; i++)
            sub_nums[i] = nums[i];

        // broadcast
        for(to = rank + 1; to < size; to++) {
            int buf_len;
            if(to != size - 1) 
                buf_len = ELEM_PER_PROC;
            else 
                buf_len = n / size + n % size;
            
            MPI_Isend(&nums[to * ELEM_PER_PROC], buf_len, MPI_INT, to, 0, MPI_COMM_WORLD, &reqs[to-1]);
        }
        MPI_Waitall(size - 1, reqs, stats);

    } else if(rank < n) {
        MPI_Request req;
        MPI_Status  stat;

        MPI_Irecv(sub_nums, ELEM_PER_PROC, MPI_INT, 0, 0, MPI_COMM_WORLD, &req);
        MPI_Wait(&req, &stat);
    }

    int partial_collapse = 0;
    for(i = 0; i < ELEM_PER_PROC; i++)
        partial_collapse += sub_nums[i];
    partial_collapse = ultimate_collapse(partial_collapse);
    printf("rank[%d]: %d\n", rank, partial_collapse);

    int *partial_collapses = NULL;
    if(rank == 0) {
        partial_collapses = (int *) malloc(sizeof(int) * comm_size);
        assert(partial_collapses != NULL);
    }
    
    // master receives all the partial collapses from slaves
    MPI_Gather( &partial_collapse, 1, MPI_INT, partial_collapses, 1, MPI_INT, 0, MPI_COMM_WORLD );


    if(rank == 0) {
        int global_collapse = 0;
        for(i = 0; i < comm_size; i++)
            global_collapse += partial_collapses[i];

        // calculate the ultimate collapse of partial collapses
        printf("ultimate collapse : %d\n", ultimate_collapse(global_collapse));
    }

    MPI_Finalize();

    return 0;
}
