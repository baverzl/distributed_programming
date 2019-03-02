#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <arpa/inet.h>
#include <dirent.h>
#include <ctype.h>
#include <time.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <pthread.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/epoll.h>

// STL
#include <queue>

#define SERVER_NAME     "tp_httpd"
#define SERVER_URL      "http://www.acme.com/software/tp_httpd/"
#define PROTOCOL        "HTTP/1.0"
#define RFC1123FMT      "%a, %d %b %Y %H:%M:%S GMT"
#define DEFAULT_PATH    "/var/tmp/cse20111599"

#define EPOLL_SIZE      1024

using namespace std;

pthread_mutex_t conn_queue_mutex;

extern ssize_t read_line(int fd, void *buffer, size_t n);

static void error_handling(const char *msg);
static void file_details( int clnt_sock, char* dir, char* name );
static void send_error( int clnt_sock, int status, const char* title, char* extra_header, const char* text );
static void send_headers( int clnt_sock, int status, const char* title, char* extra_header, const char* mime_type, off_t length, time_t mod );
static const char* get_mime_type( char* name );
static void strdecode( char* to, char* from );
static void strencode( char* to, size_t tosize, const char* from );
static int hexit( char c );
static void setnonblockingmode(int fd);

class HTTPRequest {
private:
    int conn;
    char *method;
    char *path;
    char *protocol;
public:
    HTTPRequest() { }
    HTTPRequest(int _connection, char *method, char *path, char *protocol) {
        conn = _connection;
        setMethod(method);
        setPath(path);
        setProtocol(protocol);
    }
    ~HTTPRequest() {
        if(method)   free(method);
        if(path)     free(path);
        if(protocol) free(protocol);
    }
    HTTPRequest &setMethod(char *method) {
        this->method = new char[strlen(method) + 1];
        strncpy(this->method, method, strlen(method));
        this->method[strlen(method)] = 0;
        return *this;
    }
    HTTPRequest &setPath(char *path) {
        this->path = new char[strlen(path) + 1];
        strncpy(this->path, path, strlen(path));
        this->path[strlen(path)] = 0;
        return *this;
    }
    HTTPRequest &setProtocol(char *protocol) {
        this->protocol = new char[strlen(protocol) + 1];
        strncpy(this->protocol, protocol, strlen(protocol));
        this->protocol[strlen(protocol)] = 0;
        return *this;
    }
    char *getMethod() {
        return method;
    }
    char *getPath() {
        return path;
    }
    char *getProtocol() {
        return protocol;
    }
    int from() {
        return conn;
    }
};

queue<HTTPRequest *> connQ;

void *http_request_handler(void *param)
{
    bool taskAvailable = false;
    int clnt_sock;
    int len, ich, n;
    int i;

    char *file;
    char temp[512];
    char buffer[512], method[512], path[512], protocol[512];
    char location[512], idx[512];

    struct dirent **dl;

    FILE *fp;
    struct stat sb;

    HTTPRequest *req;

    while(1) {

        pthread_mutex_lock(&conn_queue_mutex);
        if(!connQ.empty()) {
            req = connQ.front();
            connQ.pop();
            taskAvailable = true;
        }
        pthread_mutex_unlock(&conn_queue_mutex);

        if(taskAvailable && req) {
            int clnt_sock = req->from();

            // HTTP method has to be "GET"
            if( strcasecmp( req->getMethod() , "get" ) != 0 )
            {
                send_error( clnt_sock, 501, "Not Implemented", NULL, "That method is not implemented." ); 
                goto escape; 
            }
            if( req->getPath()[0] != '/' ) {
                send_error( clnt_sock, 400, "Bad Request", NULL, "Bad filename." ); 
                goto escape;
            }

            file = &(req->getPath()[1]);
            strdecode( file, file );

            if( file[0] == '\0' )
                file = "./";
            len = strlen( file );
            if( file[0] == '/' || strcmp( file, ".." ) == 0 || strncmp( file, "../", 3 ) == 0 || strstr( file, "/../" ) != NULL || (len >= 3 && strcmp( &(file[len - 3]), "/..") == 0) )
            {
                send_error( clnt_sock, 400, "Bad Request", NULL, "Illegal filename" ); 
                goto escape;
            }

            len = snprintf(temp, sizeof(temp), "%s/%s", DEFAULT_PATH, file);
            file = temp;

            // display file or file system status
            if( stat( file, &sb ) < 0 ) 
            {
                send_error( clnt_sock, 404, "Not Found", NULL, "File not found." ); 
                goto escape;
            }

            if( S_ISDIR( sb.st_mode ) ) 
            {
                if ( file[len-1] != '/' )
                {
                    snprintf( location, sizeof(location), "Location: %s/", req->getPath() );
                    send_error( clnt_sock, 302, "Found", location, "Directories must end with a slash." );
                    goto escape;
                }
                snprintf( idx, sizeof(idx), "%sindex.html", file );
                if ( stat( idx, &sb ) >= 0 )
                {
                    file = idx;
                    goto do_file;
                }
                send_headers( clnt_sock, 200, "Ok", NULL, "text/html", -1, sb.st_mtime );
                snprintf(buffer, sizeof(buffer), "\
                        <!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\" \"http://www.w3.org/TR/html4/loose.dtd\">\n\
                        <html>\n\
                        <head>\n\
                        <meta http-equiv=\"Content-type\" content=\"text/html;charset=UTF-8\">\n\
                        <title>Index of %s</title>\n\
                        </head>\n\
                        <body bgcolor=\"#99cc99\">\n\
                        <h4>Index of %s</h4>\n\
                        <pre>\n", file, file );
                write( clnt_sock, buffer, strlen(buffer) );
                n = scandir( file, &dl, NULL, alphasort );
                if ( n < 0 )
                    perror( "scandir" );
                else
                    for ( i = 0; i < n; ++i )
                        file_details( clnt_sock, file, dl[i]->d_name );

                snprintf(buffer, sizeof(buffer), "\
                        </pre>\n\
                        <hr>\n\
                        <address><a href=\"%s\">%s</a></address>\n\
                        </body>\n\
                        </html>\n", SERVER_URL, SERVER_NAME );
                write( clnt_sock, buffer, strlen(buffer) );
            } else {
do_file:
                fp = fopen( file, "r" );
                if( fp == NULL ) {
                    send_error( clnt_sock, 403, "Forbidden", NULL, "File is protected." ); 
                    printf("%s\n", strerror(errno));
                    goto escape;
                }
                send_headers( clnt_sock, 200, "Ok", NULL, get_mime_type( file ), sb.st_size, sb.st_mtime );
                while( ( ich = getc( fp ) ) != EOF )
                    write( clnt_sock, &ich, 1 );

                fclose(fp);
            }
escape:
//            close(clnt_sock);

            if(req)     free(req);

            taskAvailable = false;
        } // if(taskAvailable)
    } // while(1)
}

int main(int argc, char *argv[])
{
    int serv_sock, clnt_sock;
    struct sockaddr_in serv_adr, clnt_adr;

    if(argc != 3) {
        printf("Usage: %s <port> <number of worker threads>\n", argv[0]);
        exit(-1);
    }

    pthread_mutex_init(&conn_queue_mutex, NULL);

    serv_sock = socket(PF_INET, SOCK_STREAM, 0);

    memset(&serv_adr, 0, sizeof(serv_adr));
    serv_adr.sin_family = AF_INET; 
    serv_adr.sin_addr.s_addr = htonl(INADDR_ANY);
    serv_adr.sin_port = htons(atoi(argv[1]));

    int enable = 1;
    if (setsockopt(serv_sock, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int)) == -1) {
        perror("setsockopt");
        exit(-1);
    }

    if(bind(serv_sock, (struct sockaddr *) &serv_adr, sizeof(serv_adr)) == -1) {
        perror("bind");
        exit(-1);
    }
    if(listen(serv_sock, 10) == -1) {
        perror("listen");
        exit(-1);
    }

    int epfd = epoll_create(EPOLL_SIZE);
    struct epoll_event *ep_events = (struct epoll_event *)malloc(sizeof(struct epoll_event) * EPOLL_SIZE);
    struct epoll_event event;
    int event_cnt;

    setnonblockingmode(serv_sock);
    event.events = EPOLLIN;
    event.data.fd = serv_sock;
    epoll_ctl(epfd, EPOLL_CTL_ADD, serv_sock, &event);

    // create a thread pool
    int num_of_worker_threads = atoi(argv[2]);
    pthread_t threadPool[num_of_worker_threads];

    for(int i = 0; i < num_of_worker_threads; i++) {
        // execute thread_process_request function
        pthread_create(&threadPool[i], NULL, http_request_handler, (void *) i);
        pthread_detach(threadPool[i]);
    }

    // master work single loop
    while(1) {

        event_cnt = epoll_wait(epfd, ep_events, EPOLL_SIZE, -1);
        if(event_cnt == -1) {
            perror("epoll_wait");
            break;
        }
        printf("%d\n", event_cnt);

        for(int i = 0; i < event_cnt; i++) {
            if(ep_events[i].data.fd == serv_sock) {

                socklen_t adr_sz = sizeof(clnt_adr);

                clnt_sock = accept(serv_sock, (struct sockaddr *) &clnt_adr, &adr_sz);
                setnonblockingmode(clnt_sock);
                event.events = EPOLLIN | EPOLLET;
                event.data.fd = clnt_sock;

                // register clnt_sock to epoll instance epfd
                epoll_ctl(epfd, EPOLL_CTL_ADD, clnt_sock, &event);
                printf("Connected client IP: %s\n", inet_ntoa(clnt_adr.sin_addr));

            } else {
                char buffer[512];
                char method[512], path[512], protocol[512];

                int clnt_sock = ep_events[i].data.fd;

                int str_len;
                if ( (str_len = read_line( clnt_sock, buffer, sizeof(buffer) ) ) == -1 )
                {
                    send_error( clnt_sock, 400, "Bad Request", NULL, "No request found." );
                    break;
                }

                if(str_len == 0) {
                    epoll_ctl(epfd, EPOLL_CTL_DEL, clnt_sock, NULL);
                    close(clnt_sock);
                    printf("closed client: %d \n", clnt_sock);
                }

                if( sscanf( buffer, "%[^ ] %[^ ] %[^ ]", method, path, protocol ) != 3)
                {
                    send_error( clnt_sock, 400, "Bad Request", NULL, "Can't parse request." );
                    break;
                }

                while( read_line( clnt_sock, buffer, sizeof(buffer) ) != 0 ) {
                    if( strcmp( buffer, "\n" ) == 0 || strcmp( buffer, "\r\n" ) == 0)
                        break;
                }

                printf("%s %s %s\n", method, path, protocol);

                connQ.push( new HTTPRequest( clnt_sock, method, path, protocol ) );
            }
        }
    }

    close(serv_sock);

    return 0;
}
static void error_handling(const char *msg)
{
    fputs(msg, stderr);
    fputc('\n', stderr);
    exit(-1);
}

static void file_details( int clnt_sock, char* dir, char* name )
{
    static char encoded_name[1000];
    static char path[2000];
    struct stat sb;
    char timestr[16];
    char buffer[512];

    strencode( encoded_name, sizeof(encoded_name), name );

    snprintf( path, sizeof(path), "%s/%s", dir, name );
    if ( lstat( path, &sb ) < 0 ) {
        snprintf( buffer, sizeof(buffer), "<a href=\"%s\">%-32.32s</a>    ???\n", encoded_name, name );
        write( clnt_sock, buffer, strlen(buffer) );
    }
    else
    {
        strftime( timestr, sizeof(timestr), "%d%b%Y %H:%M", localtime( &sb.st_mtime ) );
        snprintf( buffer, sizeof(buffer), "<a href=\"%s\">%-32.32s</a>    %15s %14lld\n", encoded_name, name, timestr, (long long) sb.st_size );
        write( clnt_sock, buffer, strlen(buffer) );
    }
}


static const char* get_mime_type( char* name )
{
    char* dot;

    dot = strrchr( name, '.' );
    if ( dot == (char*) 0 )
        return "text/plain; charset=UTF-8";
    if ( strcmp( dot, ".html" ) == 0 || strcmp( dot, ".htm" ) == 0 )
        return "text/html; charset=UTF-8";
    if ( strcmp( dot, ".xhtml" ) == 0 || strcmp( dot, ".xht" ) == 0 )
        return "application/xhtml+xml; charset=UTF-8";
    if ( strcmp( dot, ".jpg" ) == 0 || strcmp( dot, ".jpeg" ) == 0 )
        return "image/jpeg";
    if ( strcmp( dot, ".gif" ) == 0 )
        return "image/gif";
    if ( strcmp( dot, ".png" ) == 0 )
        return "image/png";
    if ( strcmp( dot, ".css" ) == 0 )
        return "text/css";
    if ( strcmp( dot, ".xml" ) == 0 || strcmp( dot, ".xsl" ) == 0 )
        return "text/xml; charset=UTF-8";
    if ( strcmp( dot, ".au" ) == 0 )
        return "audio/basic";
    if ( strcmp( dot, ".wav" ) == 0 )
        return "audio/wav";
    if ( strcmp( dot, ".avi" ) == 0 )
        return "video/x-msvideo";
    if ( strcmp( dot, ".mov" ) == 0 || strcmp( dot, ".qt" ) == 0 )
        return "video/quicktime";
    if ( strcmp( dot, ".mpeg" ) == 0 || strcmp( dot, ".mpe" ) == 0 )
        return "video/mpeg";
    if ( strcmp( dot, ".vrml" ) == 0 || strcmp( dot, ".wrl" ) == 0 )
        return "model/vrml";
    if ( strcmp( dot, ".midi" ) == 0 || strcmp( dot, ".mid" ) == 0 )
        return "audio/midi";
    if ( strcmp( dot, ".mp3" ) == 0 )
        return "audio/mpeg";
    if ( strcmp( dot, ".ogg" ) == 0 )
        return "application/ogg";
    if ( strcmp( dot, ".pac" ) == 0 )
        return "application/x-ns-proxy-autoconfig";
    return "text/plain; charset=UTF-8";
}
static void send_error( int clnt_sock, int status, const char* title, char* extra_header, const char* text )
{
    char msg[1024];
    send_headers( clnt_sock, status, title, extra_header, "text/html", -1, -1 );
    
    snprintf(msg, sizeof(msg), "\
            <!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\" \"http://www.w3.org/TR/html4/loose.dtd\">\n\
            <html>\n\
            <head>\n\
            <meta http-equiv=\"Content-type\" content=\"text/html;charset=UTF-8\">\n\
            <title>%d %s</title>\n\
            </head>\n\
            <body bgcolor=\"#cc9999\">\n\
            <h4>%d %s</h4>\n", status, title, status, title );
    write( clnt_sock, msg, strlen(msg) );
    
    snprintf(msg, sizeof(msg), "%s\n", text );
    write( clnt_sock, msg, strlen(msg) );
    
    snprintf(msg, sizeof(msg), "\
            <hr>\n\
            <address><a href=\"%s\">%s</a></address>\n\
            </body>\n\
            </html>\n", SERVER_URL, SERVER_NAME );
    write( clnt_sock, msg, strlen(msg) );
}
static void send_headers( int clnt_sock, int status, const char* title, char* extra_header, const char* mime_type, off_t length, time_t mod )
{
    time_t now;
    char timebuf[100];
    char msg[1024];

    snprintf( msg, sizeof(msg), "%s %d %s\015\012", PROTOCOL, status, title );
    write( clnt_sock, msg, strlen(msg) );
    snprintf( msg, sizeof(msg), "Server: %s\015\012", SERVER_NAME );
    write( clnt_sock, msg, strlen(msg) );
    now = time( (time_t*) 0 );
    strftime( timebuf, sizeof(timebuf), RFC1123FMT, gmtime( &now ) );
    snprintf( msg, sizeof(msg), "Date: %s\015\012", timebuf );
    write( clnt_sock, msg, strlen(msg) );
    if ( extra_header ) {
        snprintf( msg, sizeof(msg), "%s\015\012", extra_header );
        write( clnt_sock, msg, strlen(msg) );
    }
    if ( mime_type ) {
        snprintf( msg, sizeof(msg), "Content-Type: %s\015\012", mime_type );
        write( clnt_sock, msg, strlen(msg) );
    }
    if ( length >= 0 ) {
        snprintf( msg, sizeof(msg), "Content-Length: %lld\015\012", (long long) length );
        write( clnt_sock, msg, strlen(msg) );
    }
    if ( mod != (time_t) -1 )
    {
        (void) strftime( timebuf, sizeof(timebuf), RFC1123FMT, gmtime( &mod ) );
        snprintf( msg, sizeof(msg), "Last-Modified: %s\015\012", timebuf );
        write( clnt_sock, msg, strlen(msg) );
    }
    snprintf( msg, sizeof(msg), "Connection: close\015\012" );
    write( clnt_sock, msg, strlen(msg) );
    snprintf( msg, sizeof(msg), "\015\012" );
    write( clnt_sock, msg, strlen(msg) );
}
static int hexit( char c )
{
    if ( c >= '0' && c <= '9' )
        return c - '0';
    if ( c >= 'a' && c <= 'f' )
        return c - 'a' + 10;
    if ( c >= 'A' && c <= 'F' )
        return c - 'A' + 10;
    return 0;       /* shouldn't happen, we're guarded by isxdigit() */
}
static void strdecode( char* to, char* from )
{
    for ( ; *from != '\0'; ++to, ++from )
    {
        if ( from[0] == '%' && isxdigit( from[1] ) && isxdigit( from[2] ) )
        {
            *to = hexit( from[1] ) * 16 + hexit( from[2] );
            from += 2;
        }
        else
            *to = *from;
    }
    *to = '\0';
}

static void strencode( char* to, size_t tosize, const char* from )
{
    int tolen;

    for ( tolen = 0; *from != '\0' && tolen + 4 < tosize; ++from )
    {
        if ( isalnum(*from) || strchr( "/_.-~", *from ) != (char*) 0 )
        {
            *to = *from;
            ++to;
            ++tolen;
        }
        else
        {
            (void) sprintf( to, "%%%02x", (int) *from & 0xff );
            to += 3;
            tolen += 3;
        }
    }
    *to = '\0';
}
void setnonblockingmode(int fd)
{
    int flag = fcntl(fd, F_GETFL, 0);
    fcntl( fd, F_SETFL, flag | O_NONBLOCK );
}
