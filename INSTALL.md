# Install

### Environment Settings

- OS: Linux, Ubuntu 20.04
- GPU: cc >= 7.0 (Ours: NVIDIA A100, NVIDIA A6000)
- CUDA version: >= 12.1
- DL Framework: PyTorch >= 2.1.0 with CUDA 12.1.

```bash
pip install torch==2.1.0 --upgrade --index-url https://download.pytorch.org/whl/cu121
```


### Clone the Project

```bash
git clone --recursive https://github.com/SiriusNEO/LLMOS-Parrot.git
```

### Install dependencies

- Step 1: Install basic requirements.

```bash
pip install -r requirements.txt
```

- Step 2: Install necessary dependencies listed in `3rdparty` folder.

```bash
cd 3rdparty/vllm
pip install -e .
```

- Step 3 (Optional): Install Optional dependencies.

(Optional) FastChat and Langchain are used only in our benchmark.

```bash
cd 3rdparty/FastChat
pip install -e ".[model_worker,webui]"
```

```bash
cd 3rdparty/langchain/libs/langchain
pip install -e .
```

<!-- (Optional) MLC-LLM is a special type of engine.

If you used MLC-LLM engines, Follow the official guide of [MLC-LLM](https://github.com/mlc-ai/mlc-llm) to install it, including the pre-compiled library and weights. The recommended commit refers to `3rdparty` folder. -->

- **! Important Notes**:

Triton 2.0.0 has some bugs in Kernel memory issues. So we enforce the version to be 2.1.0 here. You will see some dependencies warnings, but it will not affect the common usages. (The similar error also happens in [LightLLM](https://github.com/ModelTC/lightllm) kernels.)

```bash
pip install triton==2.1.0
```

### Install Parrot

(In the root folder of Parrot)

```bash
pip install -e .
```