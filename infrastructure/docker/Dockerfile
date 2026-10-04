FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# The server binds to loopback by default; inside the container it must be
# reachable from the agent container and the published port.
ENV MCP_SERVER_HOST=0.0.0.0
EXPOSE 8000
# The image serves the MCP physics server. The agent is an interactive REPL
# and needs a TTY, so docker-compose runs it as a separate service by
# overriding this command.
CMD ["python", "-m", "mcp_server.server"]
