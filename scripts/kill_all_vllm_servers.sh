#!/bin/sh
ps -ef | grep vllm | grep -v grep | awk '{print $2}' | xargs kill -9