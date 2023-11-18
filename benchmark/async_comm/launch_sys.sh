#!/bin/sh
python3 -m parrot.os.http_server --config_path ../../configs/os/localhost_os.json --log_dir log --log_filename os.log &
sleep 1
python3 -m parrot.engine.http_server --config_path vicuna-7b-v1.3-vllm.json --log_dir log --log_filename engine.log &
sleep 15