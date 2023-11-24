#!/bin/sh

echo "Start OS server ..."
python3 -m parrot.os.http_server --config_path sample_configs/os/localhost_os.json --log_dir log/ --log_filename os_single_vicuna_7b.log &

sleep 1

echo "Start one single Vicuna 7B server ..."
python3 -m parrot.engine.http_server --config_path sample_configs/engine/vicuna-7b-v1.3.json --log_dir log/ --log_filename engine_single_vicuna_7b.log &

sleep 15

echo "Successfully launched Parrot runtime system."