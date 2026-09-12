# Agents

This chapter introduces AI agents.

At the simplest level, an agent is an LLM using tools inside a repeated decision loop.

The model receives a goal, decides what action to take, observes the result, then decides what to do next.

## Core Mental Model

```text
User goal
↓
LLM decides next action
↓
tool call
↓
tool result / observation
↓
LLM examines the observation
↓
decides what to do next
↓
repeat
↓
stop when the goal is complete




                         User goal
                             ↓
                         Ministral
                             ↓
                    decide next action
                             ↓
                       tool needed?
                       /          \
                     yes           no
                      ↓             ↓
                 MCP client       finished
                      ↓
                 MCP server
                      ↓
                   tool call
                      ↓
                  observation
                      ↓
              add result to state
                      ↓
                   Ministral
                      ↓
               decide next action
                      ↓
                    repeat