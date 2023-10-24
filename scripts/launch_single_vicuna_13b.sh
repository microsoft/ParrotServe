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
python3 -m parrot.engine.native.http_server --config_path configs/engine/native/vicuna_13b_v1.3.json > log/engine_vicuna_13b.log 2>&1 &

echo "Successfully launched Parrot runtime system."