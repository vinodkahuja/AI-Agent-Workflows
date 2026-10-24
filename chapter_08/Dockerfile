# Use an official lightweight Python image
FROM python:3.11-slim

# APP_MODULE picks which chapter 8 app the container serves (override with -e APP_MODULE=...):
#   02y_image_web_app:app       image agent with a browser page (default)
#   02x_app:app                 image agent, API only
#   01y_web_chat_agent:app      web chat agent
#   01z_web_chat_mcp_agent:app  web chat agent with MCP tools
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    APP_MODULE=02y_image_web_app:app

WORKDIR /app

# Install system deps that are commonly required (adjust if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements if present and install; otherwise install minimal runtime deps
COPY requirements.txt /app/requirements.txt
RUN if [ -f /app/requirements.txt ]; then \
    pip install --no-cache-dir -r /app/requirements.txt; \
    else \
    pip install --no-cache-dir "uvicorn[standard]" fastapi; \
    fi

# Copy project files
COPY . /app

# Expose port and run uvicorn. `exec` replaces the shell with uvicorn, so Ctrl+C and
# `docker stop` reach uvicorn directly and it shuts down cleanly (without it, the
# shell swallows the signal and Docker has to force-kill the container).
EXPOSE ${PORT}
CMD ["sh", "-c", "exec uvicorn ${APP_MODULE} --host 0.0.0.0 --port ${PORT}"]
