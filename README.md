## Overview

This branch is for the OSDI'24 artifact evaluation of paper "Parrot: Efficient Serving of LLM-based Applications with Semantic Variable".

## Requirements

**Hardware Requirements.**
- 1x A100 80GB GPU.
- 4x A6000 48GB GPU.
- Both machines should have at least 50GB of free disk space to store a copy of Llama 7B weights (~13G) and a Llama 13B weights (~25G). 

**Software Requirements.**
- OS: Linux, Ubuntu 20.04.
- CUDA Version: 12.1.
- DL Framework: PyTorch 2.1.0, Triton 2.1.0.

For detailed dependencies, please refer to the `requirements.txt` file or directly use the Docker image.


## Evaluation Setup

* Artifacts Available:
The source code of Parrot is available at: https://github.com/microsoft/ParrotServe/tree/artifact

* Artifacts Functional:
Documentation: the following document includes detailed guidelines on how to build, install, test Parrot and the experiments to compare with other baselines.

* Results Reproduced:
To reproduce the main results presented in our paper, we provide a Docker image containing all the environments and baseline softwares. We also provide detailed guideline to help reproduce the results step by step.



## Environment Setup

**Clone the source code.** The source code of Parrot contains some 3rd party libraries which are organized as submodules, so remember to initialize the submodules.
```bash
git clone -b artifact git@github.com:microsoft/ParrotServe.git --recursive
cd ParrotServe
```

For installing Parrot, there are two alternatives.

#### Choice 1: Use Docker Image (Recommended)

**Launch the docker image.** To make the reproducing easier, we provide a docker image that contains all dependencies and baselines. First, build the docker image.
```bash
sudo docker build . -t parrot
```
Then start a docker instance.
```bash
TODO
```

#### Choice 2: Manual Setup

Following the instructions in [INSTALL.md](INSTALL.md) to install the dependencies and Parrot.

## Reproduce the Results

#### Artifact Overview

We put all scripts, configurations and datasets in the `artifact` folder. The `artifact` folder contains the following subfolders:
```
fastchat_scripts/  # Scripts to run FastChat baseline
figureX/           # Scripts to generate Figure X in the paper
workloads/         # Datasets used in the experiments
```

#### Reproduce Step

**Using One-click Script.** Each `figureX` folder contains a one-click script `run.sh` to reproduce the corresponding figure in the paper. For example, to reproduce `Figure 14`, you can run the following command:
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
You can manually run each step sequentially and the results of each step will be stored. Once all steps are finished, you can run the `plot.py` script to generate the final figure.

#### Experiment Details

This artifact aims to reproduce the main results presented in our paper, from `Figure 10` to `Figure 18`. The following table shows the detailed information of each experiment.

| Figure No. | Folder | Description | Approximate Running Time | Hardware | Raw Data File(s) | Generated Figure File(s) |
|------------|-------------|--------------------------|--------------------------|----------|----------------|---------------------------|
| Figure 10 | `figure10/` | Per-Decode-Latency of vLLM with varying token capacities and request rates. (Background Setting) | 1 hour | 1x A100 80GB GPU | result_fig10.txt | fig10.pdf |
| Figure 11 | `figure11/` | Average latency of single chain summary application with varying output lengths and chunk sizes. | 1 hour | 1x A100 80GB GPU | result_fig11.txt | fig11.pdf |
| Figure 12A | `figure12a/` | Average latency of chain summary applications with background requests. | 1 hour | 1x A100 80GB GPU | result_fig12.txt | fig12.pdf |
| Figure 12B | `figure12b/` | Average latency of multiple simultaneous chain summary applications. | 1 hour | 1x A100 80GB GPU | result_fig12.txt | fig12.pdf |
| Figure 13 | `figure13/` | Average latency of single map reduce summary application. | 1 hour | 1x A100 80GB GPU | result_fig13.txt | fig13.pdf |
| Figure 14 | `figure14/` | Average latency of BingCopilot applications with different batch sizes. | 1 hour | 1x A100 80GB GPU | result_fig14.txt | fig14.pdf |
| Figure 15 | `figure15/` | Latency per token of BingCopilot applications with different output lengths. | 1 hour | 1x A100 80GB GPU | result_fig15.txt | fig15.pdf |
| Figure 16 | `figure16/` | Normalized latency of serving multiple GPTs applications. | 1 hour | 1x A100 80GB GPU | result_fig16.txt | fig16.pdf |
| Figure 17 | `figure17/` | Average latency and memory usage for multi-agent programming, with varying number of files to program. | 1 hour | 1x A100 80GB GPU | result_fig17.txt | fig17.pdf |
| Figure 18 | `figure18/` | Average chat latency, map-reduce latency and chat per-decode latency in the mixed serving scenario. | 1 hour | 1x A100 80GB GPU | result_fig18.txt | fig18.pdf |

