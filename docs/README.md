# Parrot Documentation

> This repo is current a research prototype. Please open issue or contact the authors when you need help.

The documentation of Parrot, currently organized as a set of Markdown files.

## Content

- [Get Started](get_started/): In this chapter, you will learn how to install Parrot and run your first application using Parrot.
- [Documentation for Users](user_docs/): User documentation of Parrot. It contains the API specification of Parrot's OpenAI-like API, and the grammar of Parrot's frontend `pfunc`.
- [Parrot System Design](sys_design/): If you want to hack/modify Parrot, it's what you need. This chapter offers an overview of Parrot's system architecture and provides detailed explanation of Parrot's internal code organization and implementation.  
- [Version Drafts](version_drafts/): Learn about the refactor history and some of our brainstorm ideas when developing Parrot from these drafts.


## Citing Parrot

If you find Parrot useful or relevant to your research, please cite our paper as below:

```
@inproceedings{lin2024parrot,
    author = {Chaofan Lin and Zhenhua Han and Chengruidong Zhang and Yuqing Yang and Fan Yang and Chen Chen and Lili Qiu},
    title = {Parrot: Efficient Serving of LLM-based Applications with Semantic Variable},
    booktitle = {18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24)},
    year = {2024},
    address = {Santa Clara, CA},
    publisher = {USENIX Association},
    url = {https://www.usenix.org/conference/osdi24/presentation/lin-chaofan},
    month = jul
}
```