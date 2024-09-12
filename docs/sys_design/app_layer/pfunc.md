# PFunc Design and Implementation

For PFunc APIs, please refer to [the pfunc part of the user documentation](../../user_docs/pfunc.md).
In system design, we mainly dicuss the implementation of the pfunc.

## `P.function`

We can use the `@P.function` annotation to decorate a Python function and 
make it a `SemanticFunction`. This parsing process is done in `parrot/frontend/pfunc/function`.

We parse a `SemanticFunction` into several `FuncBodyPiece`, which can be a `Constant` or a `ParameterLoc`. We can bind a `Parameter` to a `SemanticVarible` by passing the SV as the argument of the function.

## Transforms

Sometimes we need to formalize / transform our prompts by some certain formats.
We abstract them as the transformations on `SemanticFunction`s. For example, the chat 
template of some chat models (e.g. Vicuna) are implemented as a `ConversationTemplate` transform.