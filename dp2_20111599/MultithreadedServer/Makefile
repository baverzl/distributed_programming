httpd:
	g++ httpd.cpp read_line.cpp -o httpd -lpthread -g
	g++ httpd_epoll.cpp read_line.cpp -o httpd_epoll -lpthread -g
	g++ http_client.cpp -o http_client -lpthread -g
	gcc echo_epollserv.c -o echo_epollserv
	gcc echo_client.c -o echo_client
clean:
	rm httpd httpd_epoll http_client echo_epollserv echo_client
