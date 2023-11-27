#!/bin/sh
python3 -m parrot.os.http_server --config_path os.json --log_dir $1 --log_filename $2 &
sleep 1
python3 -m parrot.engine.http_server --config_path engine.json --log_dir $1 --log_filename $3 --port 9001 --engine_name engine_server1 --device cuda:0 &
sleep 1
python3 -m parrot.engine.http_server --config_path engine.json --log_dir $1 --log_filename $4 --port 9002 --engine_name engine_server2 --device cuda:1 &
sleep 1
python3 -m parrot.engine.http_server --config_path engine.json --log_dir $1 --log_filename $5 --port 9003 --engine_name engine_server3 --device cuda:2 &
sleep 1
python3 -m parrot.engine.http_server --config_path engine.json --log_dir $1 --log_filename $6 --port 9004 --engine_name engine_server4 --device cuda:3 &
sleep 30