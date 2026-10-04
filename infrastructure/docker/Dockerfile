FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 8000
# The image serves the MCP physics server. The agent is an interactive REPL
# and needs a TTY, so docker-compose runs it as a separate service by
# overriding this command.
CMD ["python", "-m", "mcp_server.server"]
