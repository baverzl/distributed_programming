#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

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
    u_char **pixels;
} PPM_Gray_Image;

int readRGB_ppm(char *filename, PPM_RGB_Image *img)
{
    int i, j;
    FILE *fp;

    if((fp = fopen(filename, "rb")) == NULL) return -1;

    fscanf(fp, "%c %c\n", &img->magic[0], &img->magic[1]);
    if((img->magic[0] != 'P' || img->magic[1] != '6') && (img->magic[0] != 'P' || img->magic[1] != '3')) {
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

int horizontal_flip_ppm(PPM_RGB_Image *img)
{
    int i, j;
    for(i = 0; i < img->height; i++) {
        for(j = 0; j < img->width/2; j++) {
            PPMPixel tmp = img->pixels[i][j];
            img->pixels[i][j] = img->pixels[i][img->width - j - 1];
            img->pixels[i][img->width - j - 1] = tmp;
        }
    }
}

int rgb2grayscale_ppm(PPM_Gray_Image *dst, PPM_RGB_Image *src)
{
    int i, j;

    dst->magic[0] = src->magic[0];
    dst->magic[1] = dst->magic[1];
    dst->height   = src->height;
    dst->width    = src->width;
    dst->max      = src->max;

    dst->pixels = (u_char **) calloc(dst->height, sizeof(u_char *));
    for(i = 0; i < dst->height; i++) {
        dst->pixels[i] = (u_char *) calloc(dst->width, sizeof(u_char));
    }

    for(i = 0; i < src->height; i++) {
        for(j = 0; j < src->width; j++) {
            u_char r = src->pixels[i][j].R;
            u_char g = src->pixels[i][j].G;
            u_char b = src->pixels[i][j].B;

            dst->pixels[i][j] = ( r + g + b ) / 3;
        }
    }
}

int writeGray_ppm(char *filename, PPM_Gray_Image *img)
{
    int i, j;
    FILE *fp;

    assert(filename != NULL);

    strcat(filename, ".new");

    if((fp = fopen(filename, "w+")) == NULL) return -1;

    fprintf(fp, "P2\n");
    fprintf(fp, "%d %d\n", img->width, img->height);
    fprintf(fp, "%d\n", img->max);

    for(i = 0; i < img->height; i++) {
        for(j = 0; j < img->width; j++) {
            fprintf(fp, "%d ", img->pixels[i][j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);

    return 0;
}

int main(int argc, char **argv)
{
    PPM_RGB_Image img;
    PPM_Gray_Image new_img;

    if(argc != 2) {
        fprintf(stderr, "%s <filename>\n", argv[0]);
        return -1;
    }

    readRGB_ppm(argv[1], &img);

    horizontal_flip_ppm(&img);

    rgb2grayscale_ppm(&new_img, &img);

    writeGray_ppm(argv[1], &new_img);


    return 0;
}
