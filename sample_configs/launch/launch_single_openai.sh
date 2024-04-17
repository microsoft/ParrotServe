#!/bin/sh
echo "Start ServeCore server ..."
python3 -m parrot.serve.http_server --config_path sample_configs/core/localhost_serve_core.json --log_dir log/ --log_filename core_1_openai.log &

sleep 1

echo "Start one single Azure OpenAI server ..."
python3 -m parrot.engine.http_server --config_path sample_configs/engine/azure-openai-gpt-4.json --log_dir log/ --log_filename engine_openai.log &

sleep 3

echo "Successfully launched Parrot runtime system."