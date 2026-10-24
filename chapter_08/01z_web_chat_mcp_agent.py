# NOTE: NRP-friendly, browser-based alternative to 01_embedded_agent_speech_mcp.html.
# The original attaches MCP tools to OpenAI's Realtime voice agent. This version
# keeps the idea (a chat agent that calls MCP tools) but runs on NRP's gpt-oss and
# a typed web chat. The MCP server is notes_mcp_server.py, launched over stdio
# the same way as the chapter 3 examples. The page shows which tools were called.
#
#     cd chapter_08 && uvicorn 01z_web_chat_mcp_agent:app --port 8000
#     then open http://localhost:8000
#     try: "What time is it?"  /  "Save a note: buy milk"  /  "What notes do I have?"
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from agents import (
    Agent,
    Runner,
    SQLiteSession,
    set_default_openai_client,
    set_default_openai_api,
    set_trace_processors,
    trace,
)
from agents.mcp import MCPServerStdio, MCPServerStdioParams
from agents.tracing import TracingProcessor
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

# Load environment variables from .env file (NRP_API_KEY / NRP_BASE_URL)
load_dotenv()

# Point the Agents SDK at the NRP (Nautilus) OpenAI-compatible endpoint
client = AsyncOpenAI(
    base_url=os.getenv("NRP_BASE_URL"),
    api_key=os.getenv("NRP_API_KEY"),
)
set_default_openai_client(client, use_for_tracing=False)
set_default_openai_api("chat_completions")


class ConsoleTracingProcessor(TracingProcessor):
    """Prints trace/span utilization (tokens, duration) to the console."""

    def on_trace_start(self, trace):
        print(f"\n[trace] '{trace.name}' started")

    def on_trace_end(self, trace):
        print(f"[trace] '{trace.name}' finished")

    def on_span_start(self, span):
        pass

    def on_span_end(self, span):
        data = span.span_data.export()
        summary = f"  [span] {data.get('type', 'unknown')}"
        if data.get("name"):
            summary += f" - {data['name']}"
        if data.get("usage"):
            summary += f" | usage={data['usage']}"
        print(summary)

    def shutdown(self):
        pass

    def force_flush(self):
        pass


# Replace the default OpenAI-backend exporter with a local console printer,
# so utilization is visible without needing a real OpenAI API key
set_trace_processors([ConsoleTracingProcessor()])

CONTROLLER_MODEL = "gpt-oss"  # use "qwen3" if gpt-oss loops on tool calls
SCRIPT = str(Path(__file__).with_name("notes_mcp_server.py"))  # tool server next to this file

# ----- Conversation memory -----
# session_id -> SQLiteSession. Passing the session to Runner.run is what lets the
# agent remember earlier turns. Plain dict in memory (demo only: lost on restart).
SESSIONS: dict[str, SQLiteSession] = {}


# ----- Startup / shutdown -----
# FastAPI runs everything before `yield` at startup and everything after it at
# shutdown. We use it to start the MCP tool server once and keep it alive for all
# requests (the `async with` also stops it cleanly when the app exits).
@asynccontextmanager
async def lifespan(app: FastAPI):
    # MCPServerStdio launches notes_mcp_server.py as a child process and talks to it
    # over stdin/stdout (the same "stdio" pattern as chapter 3). The longer timeout
    # avoids the cold-start "McpError: Timed out".
    async with MCPServerStdio(
        name="Class Notes Server",
        params=MCPServerStdioParams(command="mcp", args=["run", SCRIPT]),
        client_session_timeout_seconds=90,
    ) as notes_server:
        # The agent is built here because it needs the running MCP server.
        # mcp_servers=[...] is how the agent discovers the server's tools
        # (get_current_time, add_note, list_notes) and decides when to call them.
        # Stored on app.state so the /chat endpoint can reach it.
        app.state.agent = Agent(
            name="Chat Assistant",
            instructions=(
                "You are a friendly, concise assistant. Use the notes tools to "
                "tell the time and to save or list the user's notes. "
                "Keep replies to 1-3 sentences with no markdown."
            ),
            model=CONTROLLER_MODEL,
            mcp_servers=[notes_server],
        )
        yield  # the app serves requests here, while the MCP server stays running


# ----- Web server -----
# FastAPI creates the HTTP server; uvicorn runs it. lifespan= hooks in the startup
# code above.
app = FastAPI(title="Web Chat Agent with MCP tools (NRP)", version="1.0.0", lifespan=lifespan)


# Shape of the JSON the browser sends (validated automatically by FastAPI).
class ChatIn(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str | None = None  # omit on the first message


# Shape of the JSON we send back. tools_used feeds the yellow badges on the page.
class ChatOut(BaseModel):
    reply: str
    session_id: str
    tools_used: list[str]


@app.post("/chat", response_model=ChatOut)
async def chat(body: ChatIn):
    # First message has no session_id, so create one; later messages reuse it.
    session_id = body.session_id or uuid.uuid4().hex
    # Fetch this user's history, or create an empty one the first time.
    session = SESSIONS.setdefault(session_id, SQLiteSession(session_id, db_path=":memory:"))
    print(f"[{session_id[:8]}] user: {body.message}")
    try:
        # trace(...) groups this turn's model and tool calls into one trace that the
        # ConsoleTracingProcessor prints to the terminal.
        with trace("Web chat MCP turn"):
            # The agent may call MCP tools several times before it writes the final
            # reply; Runner.run loops until it has an answer.
            result = await Runner.run(app.state.agent, body.message, session=session)
    except Exception as e:
        # 502 = "an upstream service (NRP or the MCP tool) failed".
        raise HTTPException(status_code=502, detail=f"Model or tool call failed: {e}")
    # result.new_items lists everything that happened this turn. Keep the entries
    # that are tool calls and read their tool names, so the page can show which
    # MCP tools the agent used.
    tools_used = [
        getattr(item.raw_item, "name", "?")
        for item in result.new_items
        if getattr(item, "type", None) == "tool_call_item"
    ]
    print(f"[{session_id[:8]}] tools: {tools_used} -> {result.final_output}")
    return ChatOut(reply=str(result.final_output), session_id=session_id, tools_used=tools_used)


# "New chat" button: forget this session's history.
@app.post("/reset")
async def reset(session_id: str):
    SESSIONS.pop(session_id, None)
    return {"ok": True}


# ----- The web page -----
# The whole front end is this one HTML string (HTML = structure, <style> = look,
# <script> = behavior), so there are no separate static files to deploy.
PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Chat Agent + MCP</title>
<style>
  :root { --bg:#f6f8fa; --panel:#fff; --border:#d0d7de; --text:#24292f; --muted:#57606a;
          --accent:#0969da; --user:#ddf4ff; --bot:#f0f2f4; --err:#cf222e; --tool:#fff8c5; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#0d1117; --panel:#161b22; --border:#30363d; --text:#e6edf3; --muted:#8b949e;
            --accent:#4493f8; --user:#12293f; --bot:#21262d; --err:#ff7b72; --tool:#3a3217; }
  }
  * { box-sizing: border-box; }
  body { margin:0; font:16px/1.5 system-ui,sans-serif; background:var(--bg); color:var(--text); }
  main { max-width:760px; margin:0 auto; padding:16px; display:flex; flex-direction:column; height:100vh; }
  header { display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap; }
  h1 { font-size:1.2rem; margin:0; }
  #log { flex:1; overflow-y:auto; margin:12px 0; padding:12px; background:var(--panel);
         border:1px solid var(--border); border-radius:8px; }
  .msg { padding:8px 12px; border-radius:8px; margin:8px 0; max-width:85%; white-space:pre-wrap; }
  .user { background:var(--user); margin-left:auto; }
  .bot { background:var(--bot); }
  .err { color:var(--err); }
  .tools { display:inline-block; font-size:.8rem; padding:2px 8px; margin:2px 4px 0 0;
           border-radius:10px; background:var(--tool); border:1px solid var(--border); }
  form { display:flex; gap:8px; }
  input[type=text] { flex:1; padding:10px 12px; font:inherit; color:var(--text); background:var(--panel);
                     border:1px solid var(--border); border-radius:8px; }
  button { padding:10px 16px; font:inherit; border:0; border-radius:8px; background:var(--accent);
           color:#fff; cursor:pointer; }
  button.secondary { background:transparent; color:var(--accent); border:1px solid var(--border); }
  button:disabled { opacity:.5; cursor:wait; }
</style>
</head>
<body>
<main>
  <header>
    <h1>Chat Agent + MCP tools <small style="color:var(--muted)">(gpt-oss on NRP)</small></h1>
    <button class="secondary" id="reset" type="button">New chat</button>
  </header>
  <div id="log" aria-live="polite"></div>
  <form id="form">
    <input type="text" id="msg" placeholder="Try: What time is it? / Save a note: buy milk / What notes do I have?" autocomplete="off" autofocus />
    <button id="send" type="submit">Send</button>
  </form>
</main>
<script>
  // Grab the page elements we need to read from or write to.
  const log = document.getElementById('log'), form = document.getElementById('form'),
        input = document.getElementById('msg'), send = document.getElementById('send');
  // The server gives us a session id on the first reply; we send it back every
  // turn so the agent remembers the conversation. null = no conversation yet.
  let sessionId = null;

  // Append a chat bubble ('user' or 'bot') and scroll to it. textContent (not
  // innerHTML) keeps model output from being interpreted as HTML.
  function add(text, cls) {
    const d = document.createElement('div');
    d.className = 'msg ' + cls; d.textContent = text;
    log.appendChild(d); log.scrollTop = log.scrollHeight;
    return d;
  }

  // Runs when the user presses Enter or clicks Send.
  form.addEventListener('submit', async (e) => {
    e.preventDefault();                      // stop the browser reloading the page
    const message = input.value.trim();
    if (!message) return;
    input.value = ''; send.disabled = true;  // clear the box; block double-sends
    add(message, 'user');                    // show the user's bubble immediately
    const pending = add('...', 'bot');       // placeholder until the reply arrives
    try {
      // Send the message (and session id) to our own server's /chat endpoint.
      // The API key is never here: only the server talks to NRP.
      const res = await fetch('/chat', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({message, session_id: sessionId})
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || res.statusText);
      sessionId = data.session_id;           // remember it for the next turn
      pending.textContent = data.reply;      // replace '...' with the real reply
      // Add a yellow badge for each MCP tool the agent called this turn.
      if (data.tools_used.length) {
        pending.appendChild(document.createElement('br'));
        data.tools_used.forEach((t) => {
          const s = document.createElement('span');
          s.className = 'tools'; s.textContent = 'tool: ' + t;
          pending.appendChild(s);
        });
      }
    } catch (err) {
      // Show failures (network down, NRP or tool error) in the bubble instead of silently.
      pending.textContent = 'Error: ' + err.message; pending.classList.add('err');
    } finally {
      send.disabled = false; input.focus(); log.scrollTop = log.scrollHeight;  // always re-enable
    }
  });

  // "New chat": tell the server to forget this session, then clear the screen.
  document.getElementById('reset').addEventListener('click', async () => {
    if (sessionId) await fetch('/reset?session_id=' + sessionId, {method: 'POST'});
    sessionId = null; log.innerHTML = ''; input.focus();
  });
</script>
</body>
</html>
"""


# Visiting http://localhost:8000/ returns the page above. Serving it from the same
# app means no CORS setup is needed: page and API share one origin.
@app.get("/", response_class=HTMLResponse)
async def index():
    return PAGE
