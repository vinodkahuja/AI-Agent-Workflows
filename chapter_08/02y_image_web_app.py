# NOTE: Browser front end for the free image agent (see 02x_app.py and
# chapter_07/07x_image_generation_agent_free.py).
# You type a description on a web page, the gpt-oss agent (NRP) expands it into
# a detailed prompt using the style guidelines and calls a custom tool that
# fetches the picture from Pollinations.ai (free, no API key), and the page
# displays the result. One command runs the API and the page:
#
#     cd chapter_08 && uvicorn 02y_image_web_app:app --port 8000
#     then open http://localhost:8000
import json
import os
import tempfile
import time
import urllib.parse

import requests
from agents import (
    Agent,
    Runner,
    function_tool,
    set_default_openai_client,
    set_default_openai_api,
    set_trace_processors,
    trace,
)
from agents.tracing import TracingProcessor
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
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

# ----- Style guidelines -----
# Used as the agent's instructions. They make every image share one look, and the
# last line tells the agent to turn your short request into one detailed prompt.
STYLE_GUIDELINES = """
## Style Guidelines for All Images:

- **Consistency**: Each image should maintain the same photographic quality with 3D-rendered elements seamlessly integrated
- **Color Palette**: Use a consistent scheme of blues, purples, warm golds, and greens with pops of bright accent colors
- **Lighting**: Professional photography lighting with dramatic but warm tones
- **Infographics**: Semi-transparent holographic projections that don't overwhelm the main subjects
- **Characters**: AI robots should be cute, approachable, and distinctly different from each other while maintaining a family resemblance
- **Icons**: Use universally recognizable symbols (lightbulbs for ideas, gears for processing, hearts for alignment, etc.)
- **Mood**: Optimistic, educational, and slightly futuristic without being cold or intimidating

When calling generate_image, translate these style guidelines into a single
detailed prompt string describing the specific image requested.
"""

# ----- Web server -----
# FastAPI creates the HTTP server; uvicorn runs it.
app = FastAPI(title="Image Generator Web App (NRP/free)", version="1.0.0")


# Shape of the JSON the browser sends. FastAPI validates it automatically and
# returns a 422 error if "input" is missing or empty. This is the only thing the
# caller controls; model, size and style are fixed in this file.
class GenerateIn(BaseModel):
    input: str = Field(..., min_length=1, description="Plain-language image request.")


# ----- The tool -----
# @function_tool turns this function into a tool the agent can call. The agent
# reads the docstring and type hints to learn what the tool does and what
# arguments to pass (here: one text prompt).
@function_tool
def generate_image(prompt: str) -> str:
    """Generate an image from a text prompt using a free image generation
    service and save it to a temporary file.

    Args:
        prompt: A detailed description of the image to generate.
    """
    # Pollinations.ai builds the image from the prompt in the URL (free, no API key).
    # The prompt must be URL-encoded; the time-based seed gives a new picture each call.
    encoded_prompt = urllib.parse.quote(prompt)
    seed = int(time.time())
    url = (
        f"https://image.pollinations.ai/prompt/{encoded_prompt}"
        f"?width=1024&height=1024&seed={seed}&nologo=true"
    )
    response = requests.get(url, timeout=90)  # image generation can take a while
    response.raise_for_status()               # turn HTTP errors into exceptions

    # Save the JPEG bytes to a temporary file. The tool returns the file path,
    # which is how the server finds the image after the agent finishes.
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.write(response.content)
    tmp.close()
    return tmp.name


# ----- The agent -----
def build_agent() -> Agent:
    """Fresh agent per request so no state leaks between users."""
    return Agent(
        name="Image generator",
        instructions=STYLE_GUIDELINES,
        model=CONTROLLER_MODEL,
        tools=[generate_image],
    )


# ----- Reading the agent's result -----
# result.new_items lists everything that happened during the run: the agent's
# tool calls and the tool outputs. These two helpers pull out what we need.
def extract_image_path(result) -> str | None:
    """Find the generate_image tool output (a file path) in the run result."""
    for item in getattr(result, "new_items", []) or []:
        if getattr(item, "type", None) == "tool_call_output_item" and isinstance(
            getattr(item, "output", None), str
        ):
            if os.path.exists(item.output):  # only accept a path that really exists
                return item.output
    return None


def extract_prompt(result) -> str:
    """The detailed prompt the agent wrote for the tool (shown under the image)."""
    for item in getattr(result, "new_items", []) or []:
        raw = getattr(item, "raw_item", None)
        if getattr(item, "type", None) == "tool_call_item":
            # The tool call's arguments arrive as a JSON string like {"prompt": "..."}.
            try:
                return json.loads(getattr(raw, "arguments", "{}")).get("prompt", "")
            except (TypeError, ValueError):
                pass
    return ""


# ----- The API endpoint -----
# The page's JavaScript calls this. response_class=Response means we return raw
# bytes (the JPEG) instead of JSON.
@app.post("/generate", response_class=Response)
async def generate(body: GenerateIn):
    print(f"Generating image for input: {body.input}")
    try:
        # trace(...) groups the model and tool calls of this request into one trace
        # that the ConsoleTracingProcessor prints to the terminal.
        with trace("Image generation"):
            # The agent reads the text, writes a detailed prompt, and calls
            # generate_image. Runner.run loops until the agent is done.
            result = await Runner.run(build_agent(), body.input)
    except Exception as e:
        # 502 = "an upstream service (NRP or Pollinations) failed"; the page shows it.
        raise HTTPException(status_code=502, detail=f"Agent or image service failed: {e}")

    image_path = extract_image_path(result)
    if not image_path:
        # The agent answered in text without calling the tool (gpt-oss sometimes does).
        raise HTTPException(
            status_code=500, detail="The agent did not produce an image. Try again."
        )
    # Read the image into memory, then delete the temp file: the server keeps nothing.
    with open(image_path, "rb") as f:
        jpeg_bytes = f.read()
    os.remove(image_path)

    # Prompt goes in a header (URL-quoted so it is header-safe); the body is the JPEG.
    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={"X-Image-Prompt": urllib.parse.quote(extract_prompt(result))[:4000]},
    )


# ----- The web page -----
# The whole front end is this one HTML string (HTML = structure, <style> = look,
# <script> = behavior), so there are no separate static files to deploy.
PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Image Agent</title>
<style>
  :root { --bg:#f6f8fa; --panel:#fff; --border:#d0d7de; --text:#24292f; --muted:#57606a;
          --accent:#0969da; --err:#cf222e; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#0d1117; --panel:#161b22; --border:#30363d; --text:#e6edf3; --muted:#8b949e;
            --accent:#4493f8; --err:#ff7b72; }
  }
  * { box-sizing: border-box; }
  body { margin:0; font:16px/1.5 system-ui,sans-serif; background:var(--bg); color:var(--text); }
  main { max-width:820px; margin:0 auto; padding:16px; }
  h1 { font-size:1.3rem; margin:8px 0 4px; }
  p.sub { margin:0 0 16px; color:var(--muted); }
  form { display:flex; gap:8px; flex-wrap:wrap; }
  input[type=text] { flex:1; min-width:220px; padding:10px 12px; font:inherit; color:var(--text);
                     background:var(--panel); border:1px solid var(--border); border-radius:8px; }
  button { padding:10px 18px; font:inherit; border:0; border-radius:8px; background:var(--accent);
           color:#fff; cursor:pointer; }
  button:disabled { opacity:.5; cursor:wait; }
  #out { margin-top:16px; background:var(--panel); border:1px solid var(--border);
         border-radius:8px; padding:12px; min-height:80px; }
  #out img { display:block; max-width:100%; height:auto; border-radius:6px; }
  #status { color:var(--muted); }
  #status.err { color:var(--err); }
  details { margin-top:10px; color:var(--muted); font-size:.9rem; }
  a.dl { display:inline-block; margin-top:8px; color:var(--accent); }
</style>
</head>
<body>
<main>
  <h1>Image Agent</h1>
  <p class="sub">Describe an image. The agent (gpt-oss on NRP) writes a detailed prompt and a free generator draws it.</p>
  <form id="form">
    <input type="text" id="msg" placeholder="e.g. A friendly robot teaching kids about solar energy" autocomplete="off" autofocus />
    <button id="go" type="submit">Generate</button>
  </form>
  <div id="out"><span id="status">Your image will appear here. Generation takes 10-60 seconds.</span></div>
</main>
<script>
  // Grab the page elements we need to read from or write to.
  const form = document.getElementById('form'), input = document.getElementById('msg'),
        go = document.getElementById('go'), out = document.getElementById('out');
  // Address of the previous image held in browser memory (freed before showing a new one).
  let lastUrl = null;

  // Runs when the user presses Enter or clicks Generate.
  form.addEventListener('submit', async (e) => {
    e.preventDefault();                      // stop the browser reloading the page
    const text = input.value.trim();
    if (!text) return;
    go.disabled = true;                      // block double-clicks while generating
    out.innerHTML = '<span id="status">Generating... the agent is writing a prompt and drawing it.</span>';
    try {
      // Send the text to our own server's /generate endpoint. The API key is never
      // here: only the server talks to NRP.
      const res = await fetch('/generate', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({input: text})
      });
      if (!res.ok) {
        // On failure the server sends JSON like {"detail": "..."}; show that message.
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || res.statusText);
      }
      // The agent's detailed prompt comes back in a response header (URL-encoded).
      const prompt = decodeURIComponent(res.headers.get('X-Image-Prompt') || '');
      // The response body is the raw JPEG. A "blob URL" lets <img> display it
      // straight from browser memory, with nothing saved on the server.
      const blob = await res.blob();
      if (lastUrl) URL.revokeObjectURL(lastUrl);   // release the previous image
      lastUrl = URL.createObjectURL(blob);
      out.innerHTML = '';
      const img = document.createElement('img');   // the picture itself
      img.src = lastUrl; img.alt = text;
      out.appendChild(img);
      const dl = document.createElement('a');      // saves it to your Downloads folder
      dl.className = 'dl'; dl.href = lastUrl; dl.download = 'image.jpg'; dl.textContent = 'Download image';
      out.appendChild(dl);
      // Expandable section showing how the agent rewrote the user's short text.
      if (prompt) {
        const d = document.createElement('details');
        const s = document.createElement('summary'); s.textContent = 'Prompt the agent wrote';
        const p = document.createElement('p'); p.textContent = prompt;
        d.append(s, p); out.appendChild(d);
      }
    } catch (err) {
      // Show failures (network down, NRP or Pollinations error) on the page.
      out.innerHTML = '';
      const s = document.createElement('span');
      s.id = 'status'; s.className = 'err'; s.textContent = 'Error: ' + err.message;
      out.appendChild(s);
    } finally {
      go.disabled = false; input.focus();  // always re-enable the button
    }
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
