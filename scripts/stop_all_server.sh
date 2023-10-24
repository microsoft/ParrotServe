#!/bin/sh
echo "Stop all Parrot servers ..."
ps -ef | grep http_server | grep -v grep | awk '{print $2}' | xargs kill -9
echo "Successfully killed all Parrot servers."