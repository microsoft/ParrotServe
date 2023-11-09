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
python3 -m parrot.engine.http_server --config_path configs/engine/openai/azure-openai-gpt-3.5-turbo.json > log/engine_openai.log 2>&1 &

echo "Successfully launched Parrot runtime system."