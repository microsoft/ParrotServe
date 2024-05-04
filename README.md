# Artifact of Parrot [OSDI'24]

This branch is for the OSDI'24 artifact evaluation of paper "Parrot: Efficient Serving of LLM-based Applications with Semantic Variable".

Parrot is a serving system for LLM-based Applications. It is designed as a distributed system where there is a central manager (called `os` in the code) that manages engines (which run the LLM models). Hence to launch 
Parrot, we need to start an `os_server` and multiple `engine_server`s, where they communicate with each other through localhost HTTP requests.

## 0. Requirements

**Hardware Requirements.**
- NVIDIA A100 (80GB) GPU x 1.
- NVIDIA A6000 (48GB) GPUs x 4.
- Both machines should have at least 50GB of free disk space to store a copy of Llama 7B weights (~13G) and a Llama 13B weights (~25G). 

**Software Requirements.**
- OS: Linux, Ubuntu 20.04.
- CUDA Version: 12.1.
- DL Framework: PyTorch 2.1.0, Triton 2.1.0.

For detailed dependencies, please refer to the `requirements.txt` file or directly use the Docker image.


## 1. Evaluation Setup

* Artifacts Available:
The source code of Parrot is available at: https://github.com/microsoft/ParrotServe/tree/artifact

* Artifacts Functional:
Documentation: the following document includes detailed guidelines on how to build, install, test Parrot and the experiments to compare with other baselines.

* Results Reproduced:
To reproduce the main results presented in our paper, we provide a Docker image containing all the environments and baseline softwares. We also provide detailed guideline to help reproduce the results step by step.



## 2. Environment Setup

> **NOTE:** For OSDI'24 artifact evaluation committee, please directly go to Section 3 to reproduce the results on our provided servers. Note that for a newly connected shell, 2.5 may still need to be executed.

### 2.1. Clone the Source Code
The source code of Parrot contains some 3rd party libraries which are organized as submodules, so remember to initialize the submodules.
```bash
git clone -b artifact git@github.com:microsoft/ParrotServe.git --recursive
cd ParrotServe
```

### 2.2. Install Parrot Library

Parrot has been wrapped as a Python library. For installing Parrot, there are two alternatives.

* Choice 1: Use Docker Image (Recommended). To make the reproducing easier, we provide a docker image that contains all dependencies and baselines. First, build the docker image. The building procedure usually takes 8~10 minutes.
    ```bash
    sudo docker build . -t parrot_artifact
    ```

    Then start a docker instance (Run the following command in the root directory of this repo).
    ```bash
    docker run --gpus all --shm-size=50gb -itd -v $PWD/../ParrotServe:/app --name parrot_artifact parrot_artifact /bin/bash
    docker exec -it parrot_artifact /bin/bash
    ```

* Choice 2: Manual Setup. Following the instructions in [INSTALL.md](INSTALL.md) to install the dependencies and Parrot.

### 2.3. Prepare the Dataset 

Most of our datasets are already included in the repository (in the `artifact/workloads/` folder). The ShareGPT dataset (For benchmarking chat applications) needs to be downloaded manually due to its large size. 
```bash
cd artifact/workloads/sharegpt
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

### 2.4. Prepare the Models

Our experiments use two LLM models: Llama 7B and Llama 13B. The weights of these models are not included in the repository due to their large size. We provide a script to download the weights of these models.
```bash
cd artifact
bash load_models.sh 7b # load Llama 7B weights
bash load_models.sh 13b # load Llama 13B weights
```
For A6000*4 experiments, we only need to load Llama 7B weights.

### 2.5. Set the Environment Variables

Before running the experiments, remember to set the necessary environment variables (like on/off simulated network latency).
```bash
source artifact/eval_env.sh
```

## 3. Reproduce the Results

This artifact will validate the evaluation results in the paper draft (`Figure11` - `Figure18`), including Parrot and baseline's performance on Data Analytics on Long Documents, Serving Popular LLM Applications, Multi-agent Applications, and Scheduling of Mixed Workloads, to demonstrate the claims in our paper that Parrot can efficiently optimize LLM-based applications serving by adopting Semantic Variable to uncover correlations between different LLM requests.

### 3.1. Artifact Overview

We put all scripts, configurations and datasets in the `artifact/` folder. The `artifact/` folder contains the following subfolders:
```
fastchat_scripts/  # Scripts to run FastChat baseline
figureX/           # Scripts to generate Figure X in the paper
model_loader_scripts/  # Scripts to download LLMs
workloads/         # Datasets used in the experiments
```

### 3.2. Reproduce Step

**Using One-click Script.** Each `figureX` folder contains a one-click script `run.sh` to reproduce the corresponding figure in the paper. For example, to reproduce `Figure 14`, you can run the following command (in the root directory of this repo):
```bash
cd artifact/figure14
bash run.sh
```
The generated figure can be found directly in current folder named `fig14.pdf` and the raw data is stored in files with names like `result_xxx.txt`.

**Splitting Time-consuming Experiments.** Some experiments are time-consuming and have a risk of being interrupted. Instead of using one-click scripts, you can also split the `run.sh` into serveral parts and run them separately. A `run.sh` script is usually written as follows:
```bash
#!/bin/sh

# Step 1
bash xxx

# Step 2
bash xxx

# Plot the results
python3 plot.py
```
You can manually run each step sequentially and the result of each step will be stored. Once all steps are finished, you can run the `plot.py` script to generate the final figure.

### 3.3. Experiment Details

The following table shows the detailed information of each experiment.

| Figure No. | Folder | Description | Expected Running Time | Hardware | Models | Raw Data File(s) | Generated Figure File(s) |
|------------|-------------|--------------------------|--------------------------|----------|----------|----------------|---------------------------|
| Figure 11 | `figure11/` | Average latency of single chain summary application with varying output lengths and chunk sizes. | **7 hours** | NVIDIA A100 (80GB) GPU x 1 | Llama 13B | `result_hf_olen.txt`, `result_hf_csize.txt`, `result_vllm_olen.txt`, `result_vllm_csize.txt`, `result_parrot_olen.txt`, `result_parrot_csize.txt` | `fig11_a.pdf`, `fig11_a.pdf` |
| Figure 12A | `figure12a/` | Average latency of chain summary applications with background requests. | **2 hours** | NVIDIA A100 (80GB) GPU x 1 | Llama 13B | `result_vllm.txt`, `result_parrot.txt` | `fig12_a.pdf` |
| Figure 12B | `figure12b/` | Average latency of multiple simultaneous chain summary applications. | 30 min | NVIDIA A100 (80GB) GPU x 1 | Llama 13B | `result_vllm.txt`, `result_parrot.txt` | `fig12_b.pdf` |
| Figure 13 | `figure13/` | Average latency of single map reduce summary application. | 30 min | NVIDIA A100 (80GB) GPU x 1 | Llama 13B | `result_vllm_olen.txt`, `result_vllm_csize.txt`, `result_parrot_olen.txt`, `result_parrot_csize.txt` | `fig13_a.pdf`, `fig13_b.pdf` |
| Figure 14 | `figure14/` | Average latency of BingCopilot applications with different batch sizes. | 20 min | NVIDIA A100 (80GB) GPU x 1 | Llama 7B | `result.txt` | `fig14.pdf` |
| Figure 15 | `figure15/` | Latency per token of BingCopilot applications with different output lengths. | 40 min | NVIDIA A100 (80GB) GPU x 1 | Llama 7B | `result_32.txt`, `result_64.txt` | `fig15_a.pdf`, `fig15_b.pdf` |
| Figure 16 | `figure16/` | Normalized latency of serving multiple GPTs applications. | 1 hour 15 min | NVIDIA A6000 (48GB) GPUs x 4 | Llama 7B | `result_vllm.txt`, `result_parrot_paged.txt`, `result_parrot.txt` | `fig16.pdf` |
| Figure 17 | `figure17/` | Average latency and memory usage for multi-agent programming, with varying number of files to program. | **3 hours** | NVIDIA A100 (80GB) GPU x 1 | Llama 13B | `result_vllm_lat.txt`, `result_vllm_thr.txt`, `result_parrot_paged.txt`, `result_parrot_no_share.txt`, `result_parrot.txt` | `fig17_a.pdf`, `fig17_b.pdf` |
| Figure 18 | `figure18/` | Average chat latency, map-reduce latency and chat per-decode latency in the mixed serving scenario. | 15 min | NVIDIA A6000 (48GB) GPUs x 4 | Llama 7B | `result_vllm_lat.txt`, `result_vllm_thr.txt`, `result_parrot.txt` | `fig18.pdf` |


