# Parrot: High Performance Runtime System for Semantic Programming​

This project is a research prototype for now. Being eargerly iterated.

### Install

**Install dependencies:**

```bash
pip install -r requirements.txt
pip install triton==2.1.0
```

**Install Parrot:**

```bash
python3 setup.py develop
```

### Start a PCore Server

```bash
python3 -m parrot.os.http_server --config_path configs/os/localhost_os.json
```

### Start a Vicuna-13b Engine Server

```bash
python3 -m parrot.engine.native.http_server --config_path configs/engine/native/vicuna_13b_v1.3.json
```