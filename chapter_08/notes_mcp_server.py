# Small FastMCP tool server used by 01z_web_chat_mcp_agent.py.
# Run standalone to inspect it:  mcp dev chapter_08/notes_mcp_server.py
from datetime import datetime

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Class Notes")

# In-memory notes (demo only: cleared when the server restarts)
_notes: list[str] = []


@mcp.tool()
def get_current_time() -> str:
    """Return the current local date and time."""
    return datetime.now().strftime("%A, %Y-%m-%d %H:%M:%S")


@mcp.tool()
def add_note(text: str) -> dict:
    """Save a short note for the user."""
    _notes.append(text)
    return {"status": "saved", "count": len(_notes)}


@mcp.tool()
def list_notes() -> dict:
    """Return every note saved so far."""
    return {"count": len(_notes), "notes": list(_notes)}


if __name__ == "__main__":
    mcp.run()
