#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define IMAGE_PART_SEND_RECV   0

typedef unsigned char u_char;

typedef struct {
    u_char R;
    u_char G;
    u_char B;
} PPMPixel;

typedef struct {
    char magic[2];
    int width;
    int height;
    int max;
    PPMPixel **pixels;
} PPM_RGB_Image;

typedef struct {
    char magic[2];
    int width;
    int height;
    int max;
    u_char *pixels;
} PPM_Gray_Image;

int is_master(int rank)
{
    return (rank == 0);
}
int min(int a, int b)
{
    return (a < b)? a: b;
}

int read_rgb_ppm(char *filename, PPM_RGB_Image *img)
{
    int i, j;
    FILE *fp;

    if((fp = fopen(filename, "rb")) == NULL) return -1;

    fscanf(fp, "%c %c\n", &img->magic[0], &img->magic[1]);
    if((img->magic[0] != 'P' || img->magic[1] != '3') && (img->magic[0] != 'P' || img->magic[1] != '6')) {
        fprintf(stderr, "Invalid PPM format\n");
        return -1;
    }

    fscanf(fp, "%d %d\n", &img->width, &img->height);
    fscanf(fp, "%d\n", &img->max);

    if(img->max != 255) {
        fprintf(stderr, "Invalid image format\n");
        return -1;
    }

    // dynamic allocation
    img->pixels = (PPMPixel **) calloc(img->height, sizeof(PPMPixel *));
    for(i = 0; i < img->height; i++) {
        img->pixels[i] = (PPMPixel *) calloc(img->width, sizeof(PPMPixel));
    }

    for(i = 0; i < img->height; i++) {
        for(j = 0; j < img->width; j++) {
            fread(&img->pixels[i][j], sizeof(PPMPixel), 1, fp);
        }
    }

    fclose(fp);

    return 0;
}

int horizontal_flip_ppm(PPM_RGB_Image *img, int rank, int pool_size)
{
    int i, j;

    // node[rank] : [ partition * rank, partiion * (rank + 1) )
    int partition = img->height / pool_size + 1;

    int from = partition * rank;
    int to   = min(partition * (rank + 1), img->height);

    for(i = from; i < to; ++i) {
        for(j = 0; j < img->width / 2; ++j) {
            PPMPixel tmp = img->pixels[i][j];
            img->pixels[i][j] = img->pixels[i][img->width - j - 1];
            img->pixels[i][img->width - j - 1] = tmp;
        }
    }
}

int rgb2grayscale_ppm(PPM_Gray_Image *dst, PPM_RGB_Image *src, int rank, int pool_size)
{
    int i, j;
    
    if(src == NULL) return -1;

    dst->magic[0] = src->magic[0];
    dst->magic[1] = src->magic[1];
    dst->height   = src->height;
    dst->width    = src->width;
    dst->max      = src->max;

    // dynamic allocation
    dst->pixels = (u_char *) calloc(dst->height * dst->width, sizeof(u_char));

    int partition = src->height / pool_size + 1;

    int from = partition * rank;
    int to   = min(partition * (rank + 1), src->height);

    for(i = from; i < to; ++i) {
        for(j = 0; j < src->width; ++j) {
            u_char r = src->pixels[i][j].R;
            u_char g = src->pixels[i][j].G;
            u_char b = src->pixels[i][j].B;

            dst->pixels[i * src->width + j] = ( r + g + b ) / 3;
        }
    }

    // de-allocate src
    for(i = 0; i < src->height; i++)
        free(src->pixels[i]);
    free(src->pixels);
}

int write_gray_ppm(char *filename, PPM_Gray_Image *img)
{
    int i, j;
    FILE *fp;

    assert(filename != NULL);

    strcat(filename, ".gray");

    if((fp = fopen(filename, "w+")) == NULL) return -1;

    fprintf(fp, "P2\n");
    fprintf(fp, "%d %d\n", img->width, img->height);
    fprintf(fp, "%d\n", img->max);

    for(i = 0; i < img->height * img->width; i++) 
        fprintf(fp, "%d ", img->pixels[i]);
    fprintf(fp, "\n");

    fclose(fp);

    return 0;
}

int main(int argc, char **argv)
{
    int pool_size, rank;
    PPM_RGB_Image img;
    PPM_Gray_Image new_img;

    if( argc != 2 ) {
        fprintf(stderr, "%s <filename>\n", argv[0]);
        return -1;
    }

    MPI_Init( &argc, &argv );

    MPI_Comm_size( MPI_COMM_WORLD, &pool_size );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    MPI_Barrier( MPI_COMM_WORLD );

    double local_start, local_finish, local_elapsed, elapsed;

    local_start = MPI_Wtime();

    read_rgb_ppm(argv[1], &img);

    // calc partiion[rank]
    horizontal_flip_ppm(&img, rank, pool_size);

    // calc partition[rank]
    rgb2grayscale_ppm(&new_img, &img, rank, pool_size);

    int partition_h = new_img.height / pool_size + 1;
    int from = partition_h * rank;
    int to   = min(partition_h * (rank + 1), new_img.height);
    int partition_size   = (to - from) * new_img.width;

    if(is_master(rank))
    {
        int recvfrom, i, count, rank;
        MPI_Request *req_handles = NULL;
        MPI_Status *stat_handles = NULL;

        if(pool_size >= 2) {
            req_handles  = calloc(pool_size - 1, sizeof(MPI_Request));
            stat_handles = calloc(pool_size - 1, sizeof(MPI_Status));

            // receive all the partitions from other nodes
            for(recvfrom = 1; recvfrom < pool_size; ++recvfrom) {
                int partition_st_idx = partition_h * recvfrom * new_img.width;
                // non-blocking recv from rank recvfrom
                MPI_Irecv(&new_img.pixels[partition_st_idx], partition_size, MPI_CHAR, recvfrom, IMAGE_PART_SEND_RECV, MPI_COMM_WORLD, &req_handles[recvfrom - 1]);
            }

            MPI_Waitall(pool_size - 1, req_handles, stat_handles);

            int total_received = 0;
            for(rank = 1; rank < pool_size; rank++) {
                MPI_Get_count(&stat_handles[rank-1], MPI_CHAR, &count);
                printf("Received %d bytes from node-[%d]\n", count, rank);
                total_received += count;
            }

            printf("node-[master] received all the image partitions (total: %d bytes)\n", total_received);
        }

        // only master writes the result image
        write_gray_ppm(argv[1], &new_img);

        printf("node-[master] successfully saved a new image file\n");
        if(req_handles) {
            free(req_handles); req_handles = NULL;
        }
        if(stat_handles) {
            free(stat_handles); stat_handles = NULL;
        }
    } else {
        MPI_Request req_handle;
        MPI_Status  stat_handle;

        // send partition[rank] to master
        MPI_Isend(&new_img.pixels[from * new_img.width], partition_size, MPI_CHAR, 0, IMAGE_PART_SEND_RECV, MPI_COMM_WORLD, &req_handle);
        MPI_Wait(&req_handle, &stat_handle);

        //printf("Partition sent from node-[%d]\n", rank);
    }

    local_finish = MPI_Wtime();

    local_elapsed = local_finish - local_start;
    MPI_Reduce( &local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD ); // reduced result to rank 0 

    if( rank == 0 )
        printf("Elapsed time = %e seconds\n", elapsed);

    MPI_Finalize();

    return 0;
}
