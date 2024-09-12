# OpenAI Engine

The OpenAI Engine provides a compatible way for Parrot to interface with the OpenAI API.

In its internal implementation, the OpenAI Engine does not store the KV Cache (since it doesn't have access to it). For `Fill` requests, it directly appends the text into the corresponding `Context`. For `Generate` requests, it uses the text in the corresponding `Context` as the prompt and initiates an OpenAI API call.