# Chapter 8 — Files and How to Run Them

Activate the venv first: `source venv/bin/activate`.
`.env` in the repo root needs `NRP_API_KEY` and `NRP_BASE_URL`.
Install once: `pip install fastapi "uvicorn[standard]" requests "mcp[cli]<2.0.0"`

Key: **NRP / free** = runs on NRP (and free tools). **OpenAI** = needs an OpenAI API key (reference only).

## Two ways to run the apps

1. **Individually** — one app at a time. Each file below shows its own `uvicorn` command
   (plain Python on your machine), and the Docker section shows how to run any single app in a container.
2. **All together** — `docker compose up --build` starts three containers at once
   (`01y` chat on port 8000, `02y` image agent on 8001, `01z` MCP chat on 8002), and one
   `docker compose down` stops them. See "Run all apps together in Docker (Compose)" in the Docker section (file: `docker-compose.yml`).

## Voice

### 01_embedded_agent_speech.html — OpenAI

Browser voice agent using OpenAI's Realtime API (speech to speech). Open the file in a browser and paste an OpenAI Realtime token.

### 01_embedded_agent_speech_mcp.html — OpenAI

Same as above, with MCP tools attached. Open in a browser with an OpenAI Realtime token.

### 01x_embedded_agent_speech.py — NRP / free

Voice assistant in the terminal: Whisper (speech to text), gpt-oss (reply), Piper (text to speech). Needs a microphone and speakers.

    pip install faster-whisper piper-tts sounddevice soundfile numpy
    python chapter_08/01x_embedded_agent_speech.py

No microphone? Type instead: add `--text` (and `--mute` for no speakers).

## Web chat (typed, runs in the browser)

### 01y_web_chat_agent.py — NRP / free

Web chat page with a gpt-oss agent that remembers the conversation. Tick "Read replies aloud" for browser speech.

    cd chapter_08
    uvicorn 01y_web_chat_agent:app --port 8000

Open http://localhost:8000

In Docker (this app alone): `-e APP_MODULE=01y_web_chat_agent:app` (see "Run one app in Docker").

### 01z_web_chat_mcp_agent.py — NRP / free

Same web chat, but the agent calls MCP tools (time, save note, list notes). Yellow badges show which tool ran. Uses `notes_mcp_server.py`.

    cd chapter_08
    uvicorn 01z_web_chat_mcp_agent:app --port 8000

Open http://localhost:8000 and try: "What time is it?", "Save a note: buy milk", "What notes do I have?"

In Docker (this app alone): `-e APP_MODULE=01z_web_chat_mcp_agent:app` (see "Run one app in Docker").

### notes_mcp_server.py — NRP / free

The MCP tool server used by `01z`. It starts automatically; you do not run it yourself.

## Image generation

### 02_app.py — OpenAI

Image API using OpenAI's hosted image tool. Do not run without an OpenAI key.

### 02x_app.py — NRP / free (Pollinations.ai)

Image API only, no web page. `POST /generate` returns a JPEG.

    cd chapter_08
    uvicorn 02x_app:app --port 8000

Open http://localhost:8000/docs, expand `POST /generate`, click "Try it out", enter `{"input": "a robot watering a plant"}`, Execute. Or:

    curl -X POST http://localhost:8000/generate -H "Content-Type: application/json" 
      -d '{"input":"a robot watering a plant"}' --output image.jpg

In Docker (this app alone): `-e APP_MODULE=02x_app:app` (see "Run one app in Docker").

### 02y_image_web_app.py — NRP / free (Pollinations.ai)

Same agent as `02x` with a web page: type a description, the image shows in the browser, with the prompt the agent wrote.

    cd chapter_08
    uvicorn 02y_image_web_app:app --port 8000

Open http://localhost:8000

In Docker (this app alone): it is the default, no `APP_MODULE` needed (see "Run one app in Docker").

### 03_realtime_image_agent.html — OpenAI

Browser voice agent that can ask for images from the `/generate` API. Needs an OpenAI Realtime token.

## Other examples

### 06_idempotent_key_example.py — no external service

API proxy that caches tool results by a hash of the inputs. Send the same request twice: the second is instant with `cached: true`.

    cd chapter_08
    uvicorn 06_idempotent_key_example:app --port 8001

Open http://localhost:8001/docs and POST `/tool` with `{"name":"search","args":{"q":"x"}}` twice.

### 07_phoenix_metadata.py — free (local Phoenix)

Shows how to tag traces with session, user and metadata in Phoenix. It is a skeleton with no agent code.

    phoenix serve                              # terminal 1, UI at http://localhost:6006
    python chapter_08/07_phoenix_metadata.py   # terminal 2

Both web apps use port 8000, so stop one before starting another, or pass `--port 8001`.

## Docker — Dockerfile, docker-compose.yml, requirements.txt, .dockerignore — NRP / free

One image can serve any of the web apps. You can run **one app at a time** with `docker run`
(next section) or **all three together** with `docker compose` (the section after it).
Choose the app with the `APP_MODULE` environment variable.

| APP_MODULE                          | What it serves                             |
| ----------------------------------- | ------------------------------------------ |
| `02y_image_web_app:app` (default) | Image agent with a browser page            |
| `02x_app:app`                     | Image agent, API only (`POST /generate`) |
| `01y_web_chat_agent:app`          | Web chat agent with memory                 |
| `01z_web_chat_mcp_agent:app`      | Web chat agent that calls MCP tools        |

The voice script `01x` is not containerized: it needs a microphone and speakers.

### Run one app in Docker (individually) — Dockerfile

Run these from the `chapter_08` folder, where the Dockerfile is. Docker must be installed
and running, and `.env` must be in the repo root.

    cd chapter_08
    docker build -t chapter8-agents .

    # default app (image agent with browser page)
    docker run --rm --name chapter8 -p 8000:8000 --env-file ../.env chapter8-agents

    # a different app
    docker run --rm --name chapter8 -p 8000:8000 -e APP_MODULE=01z_web_chat_mcp_agent:app --env-file ../.env chapter8-agents

    # custom host port (the container still listens on 8000)
    docker run --rm --name chapter8 -p 9000:8000 --env-file ../.env chapter8-agents

Then open http://localhost:8000 (API docs at http://localhost:8000/docs). Press Ctrl+C to stop.

Run from the repo root instead? Use `docker build -t chapter8-agents chapter_08` and `--env-file .env`.

Test the API without the browser:

    curl -X POST http://localhost:8000/generate -H "Content-Type: application/json" 
      -d '{"input":"A friendly robot teaching kids about solar energy"}' --output image.jpg

Chat history and MCP notes are kept in memory, so they reset when the container restarts.
Container time is UTC, so the MCP app's "what time is it" answer is in UTC.

### Run all apps together in Docker (Compose) — docker-compose.yml

Starting each app by hand does not scale. Compose describes several containers in one file and
starts or stops them together. Our stack has three containers, all built from the same image:

| Service         | Container name           | App                                              | Open                  |
| --------------- | ------------------------ | ------------------------------------------------ | --------------------- |
| `web`         | `chapter8-chat-web`    | `01y_web_chat_agent` (typed web chat)          | http://localhost:8000 |
| `image_agent` | `chapter8-image-agent` | `02y_image_web_app` (image agent)              | http://localhost:8001 |
| `web_agent`   | `chapter8-mcp-agent`   | `01z_web_chat_mcp_agent` (chat with MCP tools) | http://localhost:8002 |

Open each address in its own browser tab; each port shows a different app. The logs say
"Uvicorn running on 0.0.0.0:8000" for all three because that is the port *inside* each
container (every container has its own network). Compose publishes them on different ports
on your machine: `8001->8000` in `docker ps` means host port 8001 goes to that container's port 8000.

    cd chapter_08
    docker compose up --build      # build the image and start all three
    docker compose down            # stop and remove all three (or press Ctrl+C, then down)

Add `-d` to `up` to run in the background, and `docker compose logs -f` to watch the logs.
In Docker Desktop the three containers appear together as one group named `chapter8-agents`.

- Keys are read from `../.env` (`NRP_API_KEY`, `NRP_BASE_URL`), so no extra setup is needed.
- Ports busy? Change them without editing the file, e.g. `WEB_PORT=9000 docker compose up`
  (also `IMAGE_AGENT_PORT` and `WEB_AGENT_PORT`).
- The three apps run side by side and do not call each other. In the book's original stack the
  voice web page calls the other two services; here each app is independent.
- Needs the Compose plugin. Docker Desktop includes it. On Linux with Docker Engine, check with
  `docker compose version`; if it is missing, install it (Ubuntu: `sudo apt install docker-compose-v2`).

### Viewing your containers

Docker Desktop is a free app that shows your containers, images and logs in a window.

1. Install it from https://www.docker.com/products/docker-desktop/ and start it.
   - Windows: needs WSL 2 (the installer offers to set it up). Restart if asked.
   - macOS: pick the Apple silicon or Intel download to match your Mac.
   - Linux: available for selected distributions (for example Ubuntu, Debian, Fedora).
     See https://docs.docker.com/desktop/setup/install/linux/ for the list. If yours
     is not supported, use the terminal commands or Portainer below.
2. Wait until the whale icon says Docker is running.
3. Start the app with `docker run` as shown above (`--name chapter8` makes it easy to find).
4. In Docker Desktop:
   - **Containers** tab: `chapter8` appears with a green dot while it runs. The Port(s)
     column links to `localhost:8000`, which opens the app in your browser.
   - Click the container name, then **Logs** to watch the `[trace]` lines live.
   - Use the Stop button to stop it. With `--rm`, it is removed when it stops.
   - **Images** tab: shows `chapter8-agents` after `docker build`, with its size.

No Docker Desktop? The same information is available in the terminal:

    docker ps                    # running containers
    docker ps -a                 # including stopped ones
    docker images                # images you have built
    docker logs -f chapter8      # live logs (Ctrl+C stops watching, not the container)
    docker stop chapter8         # stop the container

Or run Portainer, a free web dashboard, and open https://localhost:9443 (accept the
certificate warning and set an admin password on first visit):

    docker run -d --name portainer -p 9443:9443 
      -v /var/run/docker.sock:/var/run/docker.sock 
      -v portainer_data:/data portainer/portainer-ce:latest

Portainer can control every container on the machine, so use it only on your own computer
and remove it afterward with `docker rm -f portainer`.

### Troubleshooting

- `401` / auth errors: the container did not receive `NRP_API_KEY`; check the `--env-file` path.
- `no such file or directory` for the Dockerfile or `.env`: you are in the wrong folder; see "Run one app in Docker".
- Image request times out: Pollinations.ai is occasionally slow; retry.
- `port is already allocated` / port already in use: something else is using that port, often an
  earlier container of this image. Run `docker ps` to see what holds port 8000, then stop it with
  `docker stop <name>`. Or keep it running and use a different host port, e.g. `-p 9000:8000`
  (open http://localhost:9000). Adding `--name chapter8` to `docker run` makes containers easy to find.
- To iterate without rebuilding, mount the folder: add `-v "$(pwd)":/app`.

## Share it on the internet (localtunnel) — optional

localtunnel gives a running app a temporary public URL, so people outside your network
(for example students on their own phones or laptops) can try it. Good for a demo or proof of concept;
polished deployments use cloud container services.

Needs Node.js (`npx` comes with it). The npx tools in this repo need Node 22 LTS; check `node --version` if `npx` fails.

1. Start an app (any way shown above), for example on port 8000:

   cd chapter_08
   uvicorn 02y_image_web_app:app --port 8000
2. In a second terminal, run:

   npx localtunnel --port 8000
3. It prints a URL such as `https://mighty-ends-slide.loca.lt`. Open it; the first visit shows a
   password page (see the localtunnel documentation for the current password rule), then the app loads.
4. Use the app as before. Press Ctrl+C in the tunnel terminal to close it.

With Docker Compose, run one tunnel per port you want to share (8000, 8001, 8002), each in its own terminal.

Before you share a URL:

- The apps have no login. Anyone with the URL can use them, and every request uses your NRP key and quota (they cannot see the key).
- Share the URL only with your class and close the tunnel when the demo ends.
- The MCP chat (`01z`) keeps one shared notes list, so everyone using it sees the same notes.
- localtunnel can be slow or drop connections; keep a local copy of the demo ready as a backup.
