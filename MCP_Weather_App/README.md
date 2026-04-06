# MCP Weather App

A weather agent built with LangGraph and the Model Context Protocol (MCP). The client connects to a local MCP server that exposes weather tools, prompts, and resources, and lets you chat with a Gemini-powered agent to query live weather data.

## Architecture

Two-file setup:

- **`weather_server.py`** — FastMCP server exposing a tool, a prompt, and a resource over stdio transport
- **`mcp_client.py`** — LangGraph agent (Gemini → tools_condition → tool_node loop) with a REPL supporting slash commands

### MCP Primitives

| Type | Name | Description |
|------|------|-------------|
| Tool | `get_weather` | Fetches current weather for a location via OpenWeatherMap API |
| Prompt | `compare_weather_prompt` | Generates a structured weather comparison between two cities |
| Resource | `file://delivery_log` | Reads `delivery_log.txt` — a list of orders with delivery locations |

## Setup

**Install dependencies:**
```bash
pip install mcp langgraph langchain-mcp-adapters langchain-google-genai requests
```

**Set your API keys** in the respective files:
- `mcp_client.py` — replace `{{GOOGLE_GEMINI_API_KEY}}` with your Google Gemini API key (get it from [aistudio.google.com](https://aistudio.google.com))
- `weather_server.py` — `OPENWEATHERMAP_API_KEY` is already set (replace if needed, get one at [openweathermap.org](https://openweathermap.org))

## Running

```bash
python mcp_client.py
```

The client spawns the server as a subprocess automatically — no need to start `weather_server.py` separately.

## REPL Commands

| Command | Description |
|---------|-------------|
| `/prompts` | List available prompts |
| `/prompt <name> "arg1" "arg2"` | Run a specific prompt |
| `/resources` | List available resources |
| `/resource <uri>` | Fetch a resource, then optionally send it to the agent |
| `exit` / `quit` / `q` | Quit the agent |

### Example session

```
You: What's the weather in London?
AI: The current weather in London is overcast clouds, 12°C...

You: /prompt compare_weather_prompt "Paris" "Tokyo"
AI: Here is a comparison of the weather in Paris and Tokyo...

You: /resource file://delivery_log
What would you like to do with this resource? Get the current weather for all delivery locations
AI: Here is the current weather for all 10 delivery locations...
```
