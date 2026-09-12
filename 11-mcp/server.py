from mcp.server.mcpserver import MCPServer
from os.path import expanduser


mcp = MCPServer("Local AI Learning MCP Server")


@mcp.tool()
def multiply_numbers(a: float, b: float) -> float:
    """
    Multiply two numbers.
    """
    return a * b


@mcp.tool()
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
        frequency_mhz: Frequency in MHz.
    """

    frequency_hz = frequency_mhz * 1e6

    return capacitance * (voltage ** 2) * frequency_hz


@mcp.tool()
def read_text_file(path: str) -> str:
    """
    Read the contents of a local text file.

    Args:
        path: Path to the text file.

    Returns:
        Contents of the file.
    """

    path = expanduser(path)

    with open(path, "r") as file:
        return file.read()


@mcp.resource("notes://project")
def project_notes() -> str:
    """
    Return information about the current local AI project.
    """

    return """
Local AI Learning Project

Current chapter:
Chapter 12 - Agents

Previous chapter:
Chapter 11 - MCP

Current goal:
Learn how agents repeatedly choose tools and use observations.
"""


@mcp.resource("notes://models")
def model_notes() -> str:
    """
    Return information about models used in the local AI project.
    """

    return """
Local AI Model Notes

Preferred local chat model:
ministral-3:3b

Embedding model:
embeddinggemma

Embedding dimensions:
768
"""


if __name__ == "__main__":
    mcp.run()