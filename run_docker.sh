#!/bin/sh
docker build . -t parrot
docker run --gpus all -itd -v $PWD/../ParrotServe:/app --name parrot parrot /bin/bash
docker exec -it parrot /bin/bash