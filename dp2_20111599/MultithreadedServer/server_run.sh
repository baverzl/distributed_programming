#!/bin/sh

make clean && make && ./httpd_epoll 6060 10
#make clean && make && ./httpd 6060 10
