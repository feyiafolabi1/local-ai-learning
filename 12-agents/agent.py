import asyncio
import ollama

from mcp import Client, StdioServerParameters


MODEL = "ministral-3:3b"
MAX_STEPS = 8


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

        print("\nAvailable tools:")

        for tool in tools_result.tools:
            print(f"- {tool.name}")

        # -------------------------
        # Get user goal
        # -------------------------
        user_goal = input("\nGoal: ").strip()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI agent whose job is to complete the user's goal. "
                    "You may use the available MCP tools multiple times. "
                    "After every tool result, examine the observation and decide "
                    "what action is required next. "

                    "IMPORTANT FILE RULE: "
                    "If the user mentions a local file path or asks you to read a file, "
                    "you MUST call read_text_file before making any claim about the file. "
                    "Never guess whether a file exists. "
                    "Never claim that a file is missing, invalid, readable, or unreadable "
                    "until read_text_file has actually been called and returned an observation. "

                    "IMPORTANT MATH RULE: "
                    "You MUST use multiply_numbers for every multiplication. "
                    "Do not perform multiplication yourself. "

                    "If a tool fails, treat the error as an observation. "
                    "Use that observation to decide whether to retry, use another action, "
                    "ask the user for missing information, or explain that the goal cannot "
                    "currently be completed. "

                    "Continue using tools until every part of the user's goal is complete "
                    "or until the available observations show that the goal cannot be completed. "

                    "When finished, give the final answer without calling another tool."
                ),
            },
            {
                "role": "user",
                "content": user_goal,
            },
        ]

        # -------------------------
        # Agent loop
        # -------------------------
        for step in range(1, MAX_STEPS + 1):
            print(f"\n--- Agent step {step} ---")

            response = ollama.chat(
                model=MODEL,
                messages=messages,
                tools=ollama_tools,
            )

            if response.message.content:
                print("\nModel:")
                print(response.message.content)

            tool_calls = response.message.tool_calls or []

            # -------------------------
            # No tool call = agent stops
            # -------------------------
            if not tool_calls:
                print("\nAgent finished.")
                break

            messages.append(response.message)

            # -------------------------
            # Execute requested tools
            # -------------------------
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                arguments = tool_call.function.arguments

                print("\nAgent chose tool:")
                print(f"Tool: {tool_name}")
                print(f"Arguments: {arguments}")

                try:
                    tool_result = await client.call_tool(
                        tool_name,
                        arguments,
                    )

                    result_text = ""

                    for content_item in tool_result.content:
                        if hasattr(content_item, "text"):
                            result_text += content_item.text

                    if not result_text:
                        result_text = "Tool completed with no text output."

                except Exception as error:
                    result_text = f"Tool error: {error}"

                print(f"Observation: {result_text}")

                messages.append(
                    {
                        "role": "tool",
                        "tool_name": tool_name,
                        "content": result_text,
                    }
                )

        else:
            print(
                "\nAgent stopped because it reached "
                f"the maximum of {MAX_STEPS} steps."
            )


if __name__ == "__main__":
    asyncio.run(main())