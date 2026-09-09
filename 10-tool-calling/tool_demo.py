import ollama


MODEL = "ministral-3:3b"


def add_numbers(a: float, b: float) -> float:
    """
    Add two numbers.

    Args:
        a: First number.
        b: Second number.

    Returns:
        The sum of a and b.
    """
    return a + b


def multiply_numbers(a: float, b: float) -> float:
    """
    Multiply two numbers.

    Args:
        a: First number.
        b: Second number.

    Returns:
        The product of a and b.
    """
    return a * b


def calculate_power(
    capacitance: float,
    voltage: float,
    frequency_mhz: float,
) -> float:
    """
    Calculate dynamic power using P = C * V^2 * f.

    Args:
        capacitance: Capacitance in farads.
        voltage: Voltage in volts.
        frequency_mhz: Frequency in megahertz (MHz).

    Returns:
        Dynamic power in watts.
    """

    frequency_hz = frequency_mhz * 1e6

    return (
        capacitance
        * (voltage ** 2)
        * frequency_hz
    )


def read_text_file(path: str) -> str:
    """
    Read the contents of a local text file.

    Args:
        path: Full or user-relative path to the text file.

    Returns:
        The contents of the file.
    """

    if path.startswith("~/"):
        from os.path import expanduser
        path = expanduser(path)

    with open(path, "r") as file:
        return file.read()


AVAILABLE_FUNCTIONS = {
    "add_numbers": add_numbers,
    "multiply_numbers": multiply_numbers,
    "calculate_power": calculate_power,
    "read_text_file": read_text_file,
}


TOOLS = [
    add_numbers,
    multiply_numbers,
    calculate_power,
    read_text_file,
]


def main():
    user_question = input("\nYou: ").strip()

    messages = [
        {
            "role": "system",
            "content": (
                "You can use the tools provided by this application. "
                "Use a tool whenever it is useful or required to answer accurately. "
                "If the user asks about a local file and provides a path, "
                "you MUST use the read_text_file tool. "
                "Do not say that you cannot access a file if the read_text_file "
                "tool can access it. "
                "Extract numerical values and units directly from the user's message. "
                "Do not ask the user for information they have already provided."
            ),
        },
        {
            "role": "user",
            "content": user_question,
        },
    ]

    response = ollama.chat(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
    )

    print("\nModel's first response:")

    if response.message.content:
        print(response.message.content)

    tool_calls = response.message.tool_calls or []

    if not tool_calls:
        print("\nNo tool was requested.")
        return

    messages.append(response.message)

    for tool_call in tool_calls:
        function_name = tool_call.function.name
        arguments = tool_call.function.arguments

        print("\nTool requested:")
        print(f"Function: {function_name}")
        print(f"Arguments: {arguments}")

        function_to_call = AVAILABLE_FUNCTIONS.get(
            function_name
        )

        if function_to_call is None:
            print("Unknown function.")
            continue

        try:
            result = function_to_call(**arguments)

        except Exception as error:
            result = f"Tool error: {error}"

        print(f"Tool result: {result}")

        messages.append(
            {
                "role": "tool",
                "tool_name": function_name,
                "content": str(result),
            }
        )

    final_response = ollama.chat(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
    )

    print("\nFinal answer:")
    print(final_response.message.content)


if __name__ == "__main__":
    main()