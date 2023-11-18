#!/bin/sh
echo "Start OS server ..."
python3 -m parrot.os.http_server --config_path configs/os/localhost_os.json --log_dir log/ --log_filename os_single_mlc.log &

sleep 1

echo "Start one single MLC server ..."
python3 -m parrot.engine.http_server --config_path configs/engine/mlcllm/Llama-2-13b-chat-hf-q4f16_1-vulkan.json --log_dir log/ --log_filename engine_mlc_llm.log &

sleep 10

echo "Successfully launched Parrot runtime system."