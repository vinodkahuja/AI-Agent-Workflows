# NOTE: Browser-based alternative to 01x_embedded_agent_speech.py.
# Same idea (an agent that answers you, with memory across turns) but the
# interface is a web page you type into -- no microphone, speakers, Whisper or
# Piper needed. The page is served by this same FastAPI app, so one command
# runs everything:
#
#     cd chapter_08 && uvicorn 01y_web_chat_agent:app --port 8000
#     then open http://localhost:8000
#
# The reply can optionally be read aloud with the browser's built-in speech
# synthesis (a checkbox on the page), which is free and runs in the browser.
import os
import uuid

from agents import (
    Agent,
    Runner,
    SQLiteSession,
    set_default_openai_client,
    set_default_openai_api,
    set_trace_processors,
    trace,
)
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

CONTROLLER_MODEL = "gpt-oss"  # use "qwen3" if gpt-oss misbehaves

# ----- The agent -----
# One agent object shared by all requests. It holds no conversation state itself
# (that lives in the session below), so sharing it between users is safe.
# The instructions ask for short, plain replies so they also sound fine read aloud.
agent = Agent(
    name="Chat Assistant",
    instructions="""
You are a friendly, concise assistant. Keep replies short (1-3 sentences)
since they may be read aloud. Avoid lists, markdown, or special formatting.
""",
    model=CONTROLLER_MODEL,
)

# ----- Conversation memory -----
# Maps session_id -> SQLiteSession. A SQLiteSession stores the message history, and
# passing it to Runner.run is what lets the agent remember earlier turns.
# It is a plain dict in memory (demo only: everything is lost on restart).
SESSIONS: dict[str, SQLiteSession] = {}

# ----- Web server -----
# FastAPI creates the HTTP server; uvicorn runs it.
app = FastAPI(title="Web Chat Agent (NRP)", version="1.0.0")


# Shape of the JSON the browser sends. FastAPI validates it automatically and
# returns a 422 error if "message" is missing or empty.
class ChatIn(BaseModel):
    message: str = Field(..., min_length=1, description="What the user typed.")
    session_id: str | None = Field(None, description="Omit to start a new conversation.")


# Shape of the JSON we send back.
class ChatOut(BaseModel):
    reply: str
    session_id: str


@app.post("/chat", response_model=ChatOut)
async def chat(body: ChatIn):
    # First message has no session_id, so create one; later messages reuse it.
    session_id = body.session_id or uuid.uuid4().hex
    # Fetch this user's history, or create an empty one the first time.
    session = SESSIONS.setdefault(session_id, SQLiteSession(session_id, db_path=":memory:"))
    print(f"[{session_id[:8]}] user: {body.message}")
    try:
        # trace(...) groups the model calls of this turn into one trace, which the
        # ConsoleTracingProcessor above prints to the terminal.
        with trace("Web chat turn"):
            # Send the message to gpt-oss on NRP, with the history attached.
            result = await Runner.run(agent, body.message, session=session)
    except Exception as e:
        # 502 = "the upstream service (NRP) failed"; the page shows this message.
        raise HTTPException(status_code=502, detail=f"Model call failed: {e}")
    print(f"[{session_id[:8]}] assistant: {result.final_output}")
    return ChatOut(reply=str(result.final_output), session_id=session_id)


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
<title>Chat Agent</title>
<style>
  :root { --bg:#f6f8fa; --panel:#fff; --border:#d0d7de; --text:#24292f; --muted:#57606a;
          --accent:#0969da; --user:#ddf4ff; --bot:#f0f2f4; --err:#cf222e; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#0d1117; --panel:#161b22; --border:#30363d; --text:#e6edf3; --muted:#8b949e;
            --accent:#4493f8; --user:#12293f; --bot:#21262d; --err:#ff7b72; }
  }
  * { box-sizing: border-box; }
  body { margin:0; font:16px/1.5 system-ui,sans-serif; background:var(--bg); color:var(--text); }
  main { max-width:760px; margin:0 auto; padding:16px; display:flex; flex-direction:column; height:100vh; }
  header { display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap; }
  h1 { font-size:1.2rem; margin:0; }
  label { color:var(--muted); font-size:.9rem; }
  #log { flex:1; overflow-y:auto; margin:12px 0; padding:12px; background:var(--panel);
         border:1px solid var(--border); border-radius:8px; }
  .msg { padding:8px 12px; border-radius:8px; margin:8px 0; max-width:85%; white-space:pre-wrap; }
  .user { background:var(--user); margin-left:auto; }
  .bot { background:var(--bot); }
  .err { color:var(--err); }
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
    <h1>Chat Agent <small style="color:var(--muted)">(gpt-oss on NRP)</small></h1>
    <div>
      <label><input type="checkbox" id="speak" /> Read replies aloud</label>
      <button class="secondary" id="reset" type="button">New chat</button>
    </div>
  </header>
  <div id="log" aria-live="polite"></div>
  <form id="form">
    <input type="text" id="msg" placeholder="Type a message and press Enter" autocomplete="off" autofocus />
    <button id="send" type="submit">Send</button>
  </form>
</main>
<script>
  // Grab the page elements we need to read from or write to.
  const log = document.getElementById('log'), form = document.getElementById('form'),
        input = document.getElementById('msg'), send = document.getElementById('send'),
        speak = document.getElementById('speak');
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
      // Optional: read the reply aloud with the browser's built-in voice.
      if (speak.checked && 'speechSynthesis' in window) {
        speechSynthesis.cancel();
        speechSynthesis.speak(new SpeechSynthesisUtterance(data.reply));
      }
    } catch (err) {
      // Show failures (network down, NRP error) in the bubble instead of silently.
      pending.textContent = 'Error: ' + err.message; pending.classList.add('err');
    } finally {
      send.disabled = false; input.focus();  // always re-enable the form
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
