#!/bin/sh

echo "Start ServeCore server ..."
python3 -m parrot.serve.http_server --config_path sample_configs/core/localhost_serve_core.json --log_dir log/ --log_filename core_1_vicuna_13b.log &

sleep 1

echo "Start one single Vicuna 13B server ..."
python3 -m parrot.engine.http_server --config_path sample_configs/engine/vicuna-13b-v1.3.json --log_dir log/ --log_filename engine_1_vicuna_13b.log &

sleep 15

echo "Successfully launched Parrot runtime system."