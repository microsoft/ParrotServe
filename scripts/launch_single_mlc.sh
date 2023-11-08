#!/bin/sh
if [ ! -d "log/" ];then
    mkdir log
    echo "Create log folder ..."
else
    echo "Log folder already exists."
fi

echo "Start OS server ..."
python3 -m parrot.os.http_server --config_path configs/os/localhost_os.json > log/os.log 2>&1 &

echo "Start one single Vicuna 13B server ..."
python3 -m parrot.engine.http_server --config_path configs/engine/mlcllm/Llama-2-13b-chat-hf-q4f16_1-vulkan.json > log/engine_mlc_llm.log 2>&1 &

echo "Successfully launched Parrot runtime system."