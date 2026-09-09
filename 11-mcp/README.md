# MCP — Model Context Protocol

This chapter introduces MCP: Model Context Protocol.

MCP provides a standardized way for AI applications to discover and use external tools, resources, and other capabilities.

The key idea is that instead of hardcoding every tool directly into one application, capabilities can be exposed by an MCP server and discovered by an MCP client.

## Core Mental Model

```text
                    MCP server
                   /          \
               tools         resources
                 |               |
           executable        contextual
           capabilities        data
                 \               /
                  \             /
                   MCP client
                        ↓
                      LLM
                        ↓
                   final answer