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

### Start a Vicuna-13b Backend Server

```bash
python3 -m parrot.backend.native.http_server --config_path configs/backend/native/vicuna_13b_v1.3.json
```