#!/bin/sh
set -e
echo "Stop all Parrot servers ..."
ps -ef | grep parrot | grep -v grep | awk '{print $2}' | xargs kill -9
echo "Successfully killed all Parrot servers."