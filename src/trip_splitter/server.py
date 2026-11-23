# server.py
from fastmcp import FastMCP

mcp = FastMCP("My MCP Server")

@mcp.tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

@mcp.tool
def greet(name: str) -> str:
    """Return a greeting."""
    return f"Hello, {name}!"

def main() -> None:
    mcp.run()

if __name__ == "__main__":
    main()