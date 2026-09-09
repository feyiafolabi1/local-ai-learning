# Tool Calling

This chapter introduces tool calling: giving an LLM access to external functions that it can choose to use when answering a user request.

The key idea is that the LLM does not directly execute the action.

Instead, it decides:

- whether a tool is needed,
- which tool to use,
- what arguments to pass,
- then the surrounding application executes the function.

## Core Mental Model

```text
User request
↓
LLM sees available tool descriptions
↓
LLM decides whether a tool is needed
↓
if yes:
    choose function
    extract structured arguments
↓
Python executes the function
↓
tool result goes back to LLM
↓
LLM produces final response