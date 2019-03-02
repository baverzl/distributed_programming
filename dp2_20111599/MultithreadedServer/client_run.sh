#!/bin/sh

make clean && make
./http_client 127.0.0.1 6060 20 100 static_files.txt
