#!/bin/sh
ps -ef | grep fastchat | grep -v grep | awk '{print $2}' | xargs kill -9