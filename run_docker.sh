#!/bin/sh
docker build . -t parrot_artifact
docker run --gpus all --shm-size=50gb -itd -v $PWD/../ParrotServe:/app --name parrot_artifact parrot_artifact /bin/bash
docker exec -it parrot_artifact /bin/bash