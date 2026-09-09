import asyncio
import ollama

from mcp import Client, StdioServerParameters


MODEL = "ministral-3:3b"


async def main():
    server = StdioServerParameters(
        command="python3",
        args=[
            "/Users/oluwafeyisayoafolabi/local-ai-learning/11-mcp/server.py"
        ],
    )

    async with Client(server) as client:
        print("\nConnected to MCP server!")

        # -------------------------
        # Discover MCP tools
        # -------------------------
        tools_result = await client.list_tools()

        print("\nDiscovered MCP tools:")

        for tool in tools_result.tools:
            print(f"- {tool.name}")

        ollama_tools = []

        for tool in tools_result.tools:
            ollama_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.input_schema,
                    },
                }
            )

        # -------------------------
        # Discover MCP resources
        # -------------------------
        resources_result = await client.list_resources()

        print("\nDiscovered MCP resources:")

        for resource in resources_result.resources:
            print(f"- {resource.uri}")
            print(f"  Name: {resource.name}")
            print(f"  Description: {resource.description}")

        # -------------------------
        # Ask user a question
        # -------------------------
        user_question = input("\nYou: ").strip()

        # -------------------------
        # Resource router
        # -------------------------
        router_prompt = f"""
Choose which category best matches the user's question.

PROJECT
Use this for questions about:
- the current chapter
- previous chapter
- current project goal
- the local AI learning project

MODELS
Use this for questions about:
- chat models
- embedding models
- model names
- embedding dimensions

NONE
Use this when neither resource is relevant.

User question:
{user_question}

Return ONLY one word:

PROJECT
MODELS
NONE
"""

        router_response = ollama.chat(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict routing classifier. "
                        "Return only one allowed label."
                    ),
                },
                {
                    "role": "user",
                    "content": router_prompt,
                },
            ],
        )

        route = (
            router_response.message.content
            .strip()
            .upper()
            .replace("*", "")
            .replace("`", "")
        )

        # -------------------------
        # Normalize route
        # -------------------------
        if "PROJECT" in route:
            route = "PROJECT"

        elif "MODELS" in route:
            route = "MODELS"

        else:
            route = "NONE"

        resource_map = {
            "PROJECT": "notes://project",
            "MODELS": "notes://models",
            "NONE": None,
        }

        selected_resource = resource_map[route]

        print(f"\nRouter decision: {route}")
        print(
            f"Selected resource: "
            f"{selected_resource if selected_resource else 'NONE'}"
        )

        # -------------------------
        # Load selected resource
        # -------------------------
        resource_text = ""

        if selected_resource:
            resource_result = await client.read_resource(
                selected_resource
            )

            for item in resource_result.contents:
                if hasattr(item, "text"):
                    resource_text += item.text

            print("\nLoaded MCP resource:")
            print(resource_text)

        else:
            print("\nNo MCP resource loaded.")

        # -------------------------
        # Build LLM messages
        # -------------------------
        messages = [
            {
                "role": "system",
                "content": (
                    "You can use tools provided by an MCP server. "
                    "You may also receive context from MCP resources. "
                    "Use only the provided resource context for project-specific "
                    "or model-specific facts. "
                    "Do not add unsupported technical details. "
                    "Use a tool when an external action or calculation is required. "
                    "MCP stands for Model Context Protocol."
                ),
            }
        ]

        if resource_text:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        f"MCP resource: {selected_resource}\n\n"
                        f"{resource_text}"
                    ),
                }
            )

        messages.append(
            {
                "role": "user",
                "content": user_question,
            }
        )

        # -------------------------
        # First LLM call
        # -------------------------
        response = ollama.chat(
            model=MODEL,
            messages=messages,
            tools=ollama_tools,
        )

        print("\nModel's first response:")

        if response.message.content:
            print(response.message.content)

        tool_calls = response.message.tool_calls or []

        # -------------------------
        # No tool needed
        # -------------------------
        if not tool_calls:
            print("\nNo MCP tool was requested.")
            return

        messages.append(response.message)

        # -------------------------
        # Execute MCP tool calls
        # -------------------------
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = tool_call.function.arguments

            print("\nMCP tool requested:")
            print(f"Tool: {tool_name}")
            print(f"Arguments: {arguments}")

            tool_result = await client.call_tool(
                tool_name,
                arguments,
            )

            print("\nMCP tool result:")
            print(tool_result)

            result_text = ""

            for content_item in tool_result.content:
                if hasattr(content_item, "text"):
                    result_text += content_item.text

            messages.append(
                {
                    "role": "tool",
                    "tool_name": tool_name,
                    "content": result_text,
                }
            )

        # -------------------------
        # Final LLM call
        # -------------------------
        final_response = ollama.chat(
            model=MODEL,
            messages=messages,
            tools=ollama_tools,
        )

        print("\nFinal answer:")
        print(final_response.message.content)


if __name__ == "__main__":
    asyncio.run(main())