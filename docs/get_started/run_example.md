# Quick Start: Run an Example in Vicuna 13B

This tutorial will guide you step by step in actually using Parrot to serve an LLM application. We use the Vicuna 13B (Based on LLaMA 1.0 model) to run a very simple application: recommendation letter writer.

## Download the Model

Specify the model name `lmsys/vicuna-13b-v1.3` in your `Engine` config and run the `Engine`. If this is your first time to run Vicuna 13B, Parrot will automatically detect and download the model weights from HuggingFace. Optionally, you can download the weights manually and specify the model name as the path to your model in the `Engine` config.

## Launch the Server

Refer to the [previous chapter](launch_server.md).

## Run the Example

We recommend you open three terminals when testing Parrot, with 1 terminal running `ServeCore`, 1 terminal running Vicuna 13B `Engine` and 1 terminal to run the example. In the root of the project, run:

```bash
python3 examples/write_recommendation_letter.py
```

You will see the output (Ideally, it's a well-written recommend letter by a university professor!) in the terminal.