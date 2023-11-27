# Third-party Libraries

For some libraries, we have to use a specific version, or a specific branch, or a specific commit. And usually some hacks are needed to make the evaluation fair and reproducible. So we put them here (by Git submodules / modified source code).

The following libraries are referenced and hacked. For preventing embedding Git repo, we deleted `git` library in these submodules. And we maintain the original commit IDs we forked:
- [FastChat](https://github.com/lm-sys/FastChat.git): `e53c73f22efa9a37bf76af8783c96049276a2e98`
- [vLLM](https://github.com/vllm-project/vllm.git): `4b6f069b6fbb4f2ef7d4c6a62140229be61c5dd3`

For distributed inference with vLLM, we need to use Ray. The corresponding version is `2.5.1`.

All licenses are subject to the original libraries.