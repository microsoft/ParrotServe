# Parrot's Responsible AI FAQ

## What is Parrot?

Parrot is an LLM service system that optimizes the end-to-end experience of LLM-based applications. It introduces the concept of a Semantic Variable, a new abstraction that annotates a specified region in an LLM request (also known as a prompt).
Parrot allows for conventional data flow analysis to uncover the correlation across multiple LLM requests. This correlation opens a brand-new optimization space for LLM-based applications, potentially leading to significant performance improvements.

## What can Parrot do?

Parrot can optimize LLM-based applications by uncovering the correlation across multiple LLM requests. This is achieved through a new abstraction called Semantic Variable, which annotates a specified region in an LLM prompt.
By focusing on the end-to-end experience of LLM-based applications, Parrot can greatly improve the efficiency and performance of these applications.

## What is/are Parrot's intended use(s)?

Parrot is intended to optimize the inference of LLM-based applications. Its system can benefit users who rely on LLM services for various tasks, as it can improve the efficiency and speed of these services.

## How was Parrot evaluated? What metrics are used to measure performance?
Parrot's performance was evaluated through extensive evaluations on various types of LLM applications, including data analytics, chat, and AI agents. The performance of Parrot is measured primarily in terms of latency, and throughput.

## What are the limitations of Parrot? How can users minimize the impact of Parrot’s limitations when using the system?

As an inference system, the potential harmful, false or biased responses using Parrot would likely be unchanged compared to directly using the model. Thus using Parrot has no inherent benefits or risks when it comes to those types of responsible AI issues.

## What operational factors and settings allow for effective and responsible use of Parrot?

The key of Parrot’s benefit is the understanding of dataflow among multiple LLM calls. Effective use of Parrot’s APIs is necessary to expose such information for potential optimizations.
