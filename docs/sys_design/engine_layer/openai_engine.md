# OpenAI Engine

The OpenAI Engine enables seamless integration for Parrot to interact with the OpenAI API.

In its internal workings, the OpenAI Engine does not store the KV Cache, as it lacks access to it. For `Fill` requests, it appends the text directly to the corresponding `Context`. For `Generate` requests, it uses the text within the `Context` as the prompt and initiates a call to the OpenAI API.