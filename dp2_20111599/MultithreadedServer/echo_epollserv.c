#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/epoll.h>

#define BUF_SIZE 4
#define EPOLL_SIZE 50
void error_handling(char *buf);

int main(int argc, char *argv[])
{
    int serv_sock, clnt_sock;
    struct sockaddr_in serv_adr, clnt_adr;
    socklen_t adr_sz;
    int str_len, i;
    char buf[BUF_SIZE];

    struct epoll_event *ep_events;
    struct epoll_event event;
    int epfd, event_cnt;

    if(argc!=2) {
        printf("Usage : %s <port>\n", argv[0]);
        exit(1);
    }

    serv_sock=socket(PF_INET, SOCK_STREAM, 0);
    memset(&serv_adr, 0, sizeof(serv_adr));
    serv_adr.sin_family=AF_INET;
    serv_adr.sin_addr.s_addr=htonl(INADDR_ANY);
    serv_adr.sin_port=htons(atoi(argv[1]));

    if(bind(serv_sock, (struct sockaddr*) &serv_adr, sizeof(serv_adr))==-1)
        error_handling("bind() error");
    if(listen(serv_sock, 5)==-1)
        error_handling("listen() error");

    epfd = epoll_create(EPOLL_SIZE);
    ep_events = malloc(sizeof(struct epoll_event) * EPOLL_SIZE);

    event.events  = EPOLLIN; // 수신할 데이터가 존재하는 상황
    /*
     * EPOLLOUT: 출력버퍼가 비워져서 당장 데이터를 전송할 수 있는 상황
     * EPOLLPRI: OOB 데이터가 수신된 상황
     * EPOLLRDHUP: 연결이 종료되거나 Half-close가 진행된 상황, 이는 엣지 트리거 방식에서 유용하게 사용될 수 있다.
     * EPOLLERR: 에러가 발생한 상황
     * EPOLLET:  이벤트의 감지를 엣지 트리거 방식으로 동작시킨다.
     * EPOLLONESHOT: 이벤트가 한번 감지되면, 해당 파일 디스크립터에서는 더 이상 이벤트를 발생시키지 않는다.
     * 따라서 epoll_ctl 함수의 두 번째 인자로 EPOLL_CTL_MOD을 전달해서 이벤트를 재설정해야 한다.
     *
     */
    event.data.fd = serv_sock;	
    epoll_ctl(epfd, EPOLL_CTL_ADD, serv_sock, &event); 

    while(1)
    {
        // similar to select() function
        event_cnt = epoll_wait(epfd, ep_events, EPOLL_SIZE, -1);
        if(event_cnt ==- 1)
        {
            puts("epoll_wait() error");
            break;
        }

        printf("return epoll_wait\n");
        for(i = 0; i < event_cnt; i++)
        {
            if(ep_events[i].data.fd == serv_sock) // event happened on serv_sock
            {
                adr_sz = sizeof(clnt_adr);
                // accept the client!
                clnt_sock = accept(serv_sock, (struct sockaddr*)&clnt_adr, &adr_sz);
                event.events = EPOLLIN;
                event.data.fd = clnt_sock;
                // register clnt_sock file descriptor to epoll instance epfd
                epoll_ctl(epfd, EPOLL_CTL_ADD, clnt_sock, &event); // using event varialbe to observe the event
                printf("connected client: %d \n", clnt_sock);
            }
            else
            {
                str_len = read(ep_events[i].data.fd, buf, BUF_SIZE);
                if(str_len == 0)    // close request!
                {
                    // delete ep_events[i].data.fd file descriptor from epoll instance epfd
                    epoll_ctl(epfd, EPOLL_CTL_DEL, ep_events[i].data.fd, NULL);
                    close(ep_events[i].data.fd);
                    printf("closed client: %d \n", ep_events[i].data.fd);
                }
                else
                {
                    write(ep_events[i].data.fd, buf, str_len);    // echo!
                }
            }
        }
    }
    close(serv_sock);
    close(epfd);
    return 0;
}

void error_handling(char *buf)
{
    fputs(buf, stderr);
    fputc('\n', stderr);
    exit(1);
}
