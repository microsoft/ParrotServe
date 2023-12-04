# Parrot: Efficient Serving of LLM-based Application with Semantic Variables

This project is a research prototype for now. Being eargerly iterated.

![](assets/layers_arch.png)


## Install

See [INSTALL.md](INSTALL.md) for installation instructions.

## Run Parrot

**Run the Compose Script in a Single Machine**

We provide some one-click scripts to run Parrot in a single machine with sample configs. You can check them in the `sample_configs/launch` folder.

```bash
bash sample_configs/launch/launch_single_vicuna_13b.sh
```

<!-- **Run Docker Compose in a Cluster**

TODO -->

**Start an OS Server**

You can separately start an OS server.

```bash
python3 -m parrot.os.http_server --config_path <config_path>
```

**Start an Engine Server**

You can separately start an engine server. If you choose to connect to the OS server, you need to start the OS server first and specify the OS server address in the config file.

```bash
python3 -m parrot.engine.http_server --config_path <config_path>
```