#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <pthread.h>
#include <vector>

#define BUF_SIZE    100
#define NAME_SIZE   20

struct sockaddr_in serv_addr;
pthread_mutex_t mutex;

static int iterations = 0;
static int total_bytes = 0;
static int no_of_files = 0;

using namespace std;

struct File {
private:
    char *name;
public:
    File(char *name) {
        this->name = new char[strlen(name) + 1];
        strncpy(this->name, name, strlen(name));
        this->name[strlen(name)] = 0;
    }
    char *getName() {
        return name;
    }
};

vector<File *> files;

void *http_connect_handler(void * arg)
{
    int num_of_reqs_per_thread = *((int *) arg);

    char buffer[1024];

    for(int i = 0; i < num_of_reqs_per_thread; i++) 
    {
        // create a new socket to connect to a server
        int sock = socket(PF_INET, SOCK_STREAM, 0);

        pthread_mutex_lock( &mutex );
        if(connect(sock, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) == -1) {
            perror("connect");
            exit(-1);
        }
        pthread_mutex_unlock( &mutex );

        snprintf(buffer, sizeof(buffer), "GET /%s HTTP/1.1\r\n\r\n", files[ i % files.size() ]->getName() );

        puts("client http request : ");
        write( sock, buffer, strlen(buffer) );
        write( 1, buffer, strlen(buffer) );

        pthread_mutex_lock(&mutex);
        puts("client http response : ");
        int len;
        while( ( len = read( sock, buffer, sizeof(buffer) ) ) > 0) {
            write( 1, buffer, len );
            total_bytes += len;
        }
        pthread_mutex_unlock(&mutex);

        close(sock);

        pthread_mutex_lock(&mutex);
        iterations++;
        pthread_mutex_unlock(&mutex);
    }

    pthread_exit((void *)0);
}

int main(int argc, char **argv)
{
    void *thread_return;

    if(argc != 6) {
        printf("Usage: %s <ip> <port> <number of threads> <number of reqs per thread> <static_files.txt>\n", argv[0]);
        exit(1);
    }

    pthread_mutex_init( &mutex, NULL );

    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr(argv[1]);
    serv_addr.sin_port = htons(atoi(argv[2]));

    int num_of_reqs_per_thread = atoi(argv[4]);

    int num_of_threads = atoi(argv[3]);
    pthread_t tids[num_of_threads];

    char *file = argv[5];

    FILE *fp = fopen(file, "r");
    if(fp == NULL) {
        perror("fopen");
        exit(-1);
    }

    char line[256];
    char filename[256];

    while(!feof(fp)) {
        fgets(line, sizeof(line), fp);
        sscanf(line, "%s\n", filename);
        files.push_back(new File(filename));
    }

    for(int i = 0; i < num_of_threads; i++)
        pthread_create(&tids[i], NULL, http_connect_handler, (void *) &num_of_reqs_per_thread );

    for(int i = 0; i < num_of_threads; i++)
        pthread_join(tids[i], NULL);

    printf("Total iterations: %d\n", iterations);
    printf("Total %d bytes received\n", total_bytes);

    pthread_mutex_destroy(&mutex);

    return 0;
}
