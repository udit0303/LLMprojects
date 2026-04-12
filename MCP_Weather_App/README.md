# MCP Weather App

A multi-server weather and task management agent built with LangGraph and the Model Context Protocol (MCP). Supports two client modes: a single-server client (Claude) for weather only, and a multi-server client (Claude) connecting to both a weather server and a task server.

## Architecture

### Servers

| File | Name | Description |
|------|------|-------------|
| `weather_server.py` | WeatherAssistant | Weather tool, compare prompt, delivery log resource |
| `task_server.py` | TaskManagementAssistant | Task tools, trip planning prompt, meeting notes resource |

### Clients

| File | LLM | Servers |
|------|-----|---------|
| `single_server_mcp_client.py` | Claude (Anthropic) | Weather only |
| `multi_server_mcp_client.py` | Claude (Anthropic) | Weather + Tasks |

---

## MCP Primitives

### Weather Server (`weather_server.py`)

| Type | Name | Description |
|------|------|-------------|
| Tool | `get_weather` | Fetches current weather for a location via OpenWeatherMap API |
| Prompt | `compare_weather_prompt` | Structured weather comparison between two cities |
| Resource | `file://delivery_log` | Reads `delivery_log.txt` — orders with delivery locations |

### Task Server (`task_server.py`)

| Type | Name | Description |
|------|------|-------------|
| Tool | `add_task` | Adds a task to `tasks.txt` |
| Tool | `list_tasks` | Lists all tasks from `tasks.txt` |
| Tool | `remove_task` | Removes a task by partial match from `tasks.txt` |
| Prompt | `plan_trip_prompt` | Generates a day-by-day itinerary and saves it as tasks |
| Resource | `file://meeting_notes` | Reads `meeting_notes.txt` — meeting discussion points and action items |

---

## Setup

**Install dependencies:**
```bash
pip install mcp langgraph langchain-mcp-adapters langchain-anthropic requests
```

**Set your API keys:**
```bash
export ANTHROPIC_API_KEY="your_key_here"        # get it from console.anthropic.com
export OPENWEATHERMAP_API_KEY="your_key_here"   # get one at openweathermap.org
```

---

## Running

**Single-server client (weather only):**
```bash
python single_server_mcp_client.py
```

**Multi-server client (weather + tasks):**
```bash
python multi_server_mcp_client.py
```

Both clients spawn their server(s) as subprocesses automatically.

---

## REPL Commands

Both clients support the following slash commands:

| Command | Description |
|---------|-------------|
| `/prompts` | List all available prompts across all servers |
| `/prompt <server> <name> "arg1" "arg2"` | Run a specific prompt |
| `/resources` | List all available resources across all servers |
| `/resource <server> <uri>` | Fetch a resource, then optionally send it to the agent |
| `exit` / `quit` / `q` | Quit the agent |

> **Note:** For the multi-server client, `<server>` is either `weather` or `tasks`.

---

## Example Workflows

**Check weather:**
```
You: What's the weather in London?
```

**Compare weather using a prompt:**
```
You: /prompt weather compare_weather_prompt "Paris" "Tokyo"
```

**Get weather for all delivery locations:**
```
You: /resource weather file://delivery_log
What would you like to do with this resource? Get the current weather for all delivery locations
```

**Process meeting notes into tasks:**
```
You: /resource tasks file://meeting_notes
What would you like to do with this resource?       ← just press Enter
You: Process the notes and add all action items to the to-do list
```

**Plan a trip:**
```
You: /prompt tasks plan_trip_prompt "Japan" "7 days"
```

**Manage tasks:**
```
You: What's on my to-do list?
You: Remove the grocery task
```
