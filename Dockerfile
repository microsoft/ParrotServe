FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

WORKDIR /app

COPY . /install

RUN nvcc --version

RUN cd /install && pip install -r requirements.txt
RUN cd /install/3rdparty/vllm && export CUDA_HOME=/usr/local/cuda && pip install -e .
RUN cd /install/3rdparty/FastChat && pip install -e ".[model_worker,webui]"
RUN cd /install/3rdparty/langchain/libs/langchain && pip install -e .
RUN pip install triton==2.1.0
RUN cd /install && pip install -e .

RUN apt-get update
RUN apt-get install vim -y
