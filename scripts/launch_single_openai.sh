#!/bin/sh
echo "Start OS server ..."
python3 -m parrot.os.http_server --config_path sample_configs/os/localhost_os.json --log_dir log/ --log_filename os_single_openai.log &

sleep 1

echo "Start one single Azure OpenAI server ..."
python3 -m parrot.engine.http_server --config_path sample_configs/engine/azure-openai-gpt-3.5-turbo.json --log_dir log/ --log_filename engine_openai.log &

sleep 3

echo "Successfully launched Parrot runtime system."