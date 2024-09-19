# Write LLM Programs in Parrot: PFunc

PFunc (Parrot-Function Python interface) is the front-end for writing LLM programs in Parrot system. You can view it as a Pythonic wrapper of Parrot's OpenAI-like APIs w/ Semantic Variable.

(Other front-ends like [Langchain](https://www.langchain.com) and [Semantic Kernel](https://aka.ms/semantic-kernel) are not integrated into Parrot yet)

## Basic Components
- `SemanticFunction`
- `SemanticVariable`
- Placeholders (`P.Input` and `P.Output`)

## Grammar

```python
from parrot import P # 'P' is the alias of pfunc lang.

@P.function(conversation_template=P.vicuna_template)
def tell_me_a_joke(
    topic: P.Input,
    topic2: P.Input,
    joke: P.Output,
    explanation: P.Output,
):
    """Tell the me a joke about {{topic}} and {{topic2}}. {{joke}}.
    Good, then giving a short explanation to show that why it is funny.
    The explanation should be short, concise and clear. {{explanation}}.
    """

async def main():
    # Add a function call to the executor, create a session.
    joke, explanation = tell_me_a_joke(topic="Food", topic2="Cheese")

    # This will get the text from the "SemanticVariable".
    joke_content = await joke.get()
    explanation_content = await explanation.get()
```

A semantic function is defined using a `@P.function(...)` decorator. Developers can pass some necessary metadata in this decorator.

The syntax is just like the conventional definition of Python function. But there are a few differences:

- The arguments annotated by a `P.Input`, `P.Output` are considered as the `SemanticParameter`. Arguments with other annotations or no annotation are considered as `PyObject`.
- The docstring is the function definition! Plain text in the docstring are considered as the constant part. And using `{{}}` to reference function parameters, which is considered as *Placeholders*.
- When we call a semantic function, we should pass all arguments which are `P.Input` or `PyObject`. The value type of the `P.Input` argument must be a `SemanticVariable`.
- The return value of the function is a List of `SemanticVariable`s, corresponding to all `P.Output`s in the function declaration.
- Because the asynchronous design, the return values may not be immediately ready. So they are `SemanticVariable`s . If we need their contents (The content is just a string), we should use `await xx.get()` .