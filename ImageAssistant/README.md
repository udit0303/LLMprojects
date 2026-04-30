# Image Research Assistant

A multi-server MCP agent that analyzes images and researches topics on Wikipedia, served through a Gradio web UI. The agent (Claude Sonnet 4.6) chains tools from two MCP servers: vision analysis and Wikipedia search.

## Architecture

| File | Role | Description |
|------|------|-------------|
| `client.py` | LangGraph agent + Gradio UI | Claude Sonnet 4.6 with `MultiServerMCPClient` connecting to both MCP servers. Web UI for image upload + chat. |
| `visual_analysis_server.py` | MCP server (`VisualAnalysisServer`) | Single tool: `describe_image(file_path)` — loads an image from disk, sends it to Claude Haiku 4.5 as a base64 vision input, returns a one-paragraph description. |
| `wikipedia_server.py` | MCP server (`WikipediaSearch`) | Single tool: `fetch_wikipedia_info(query, num_articles)` — searches Wikipedia and returns title/summary/URL for top matches. |

The client spawns both servers as stdio subprocesses automatically. The base64 image data stays inside `visual_analysis_server.py` — only the description text crosses the LLM boundary, keeping the agent's context window lean.

---

## MCP Primitives

### `visual_analysis_server.py`

| Type | Name | Description |
|------|------|-------------|
| Tool | `describe_image` | Takes an absolute file path, base64-encodes the image server-side, calls Claude Haiku 4.5 with a landmark/object-identification prompt, returns a one-paragraph description. |

### `wikipedia_server.py`

| Type | Name | Description |
|------|------|-------------|
| Tool | `fetch_wikipedia_info` | Searches Wikipedia for a query, returns a list of `{title, summary, url}` dicts for the top N matches. Skips ambiguous and broken pages gracefully. |

---

## Setup

**Create a virtual environment and install dependencies:**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install mcp langgraph langchain-core langchain-anthropic \
            langchain-mcp-adapters anthropic wikipedia gradio \
            typing-extensions python-dotenv
```

**Configure your API key** — create a `.env` file:
```
ANTHROPIC_API_KEY=sk-ant-...    # from console.anthropic.com
```

The `.env` file is gitignored.

---

## Running

```bash
python client.py
```

The client spawns both MCP servers as stdio subprocesses, then launches a Gradio UI at `http://localhost:7860`.

---

## Using the UI

1. **Upload an image** (optional) via the image box.
2. **Type a question** in the textbox — about the image, or any general research question.
3. Click **Submit**. The agent decides which tools to call:
   - Image-only question → `describe_image`
   - Research question → `fetch_wikipedia_info`
   - "What is this and tell me about it" → chains both tools
4. Click **Quit** when done — appends a farewell, closes the Gradio server, and exits the process cleanly.

### Example flows

**Image + research (chains both tools):**
> *"What landmark is this and tell me its history?"* + upload of `giza.jpeg`
>
> Agent calls `describe_image` → "the Great Pyramid of Giza..." → calls `fetch_wikipedia_info("Great Pyramid of Giza")` → composes a final answer.

**Research only:**
> *"Tell me about the Eiffel Tower"*
>
> Agent calls `fetch_wikipedia_info` directly.

**Image only:**
> *"What's in this image?"* + upload
>
> Agent calls `describe_image` and returns the description.

---

## Files & Directories

```
ImageAssistant/
├── .env                         # your ANTHROPIC_API_KEY (gitignored, you create it)
├── .gitignore
├── client.py                    # LangGraph agent + Gradio UI
├── visual_analysis_server.py    # FastMCP vision server (Claude Haiku 4.5)
├── wikipedia_server.py          # FastMCP Wikipedia search server
├── README.md
└── venv/                        # local virtualenv
```

---

## Design Notes

- **Two Claude tiers in one app:** Sonnet 4.6 drives the agent loop (reasoning, tool selection); Haiku 4.5 handles vision (cheap, fast, sufficient for one-paragraph descriptions).
- **Server-side base64:** the previous design returned the base64 string from a `load_image_from_path` tool, then re-sent it through the LLM to a separate `get_image_description` tool — that round-tripped megabytes of text through Claude's API. Combining load + analyze into a single server-side tool keeps the image data local.
- **Graceful shutdown:** the Quit button uses a daemon thread to delay `os._exit(0)` by 1 second so the farewell message can render in the browser before the process dies.
- **Multi-server tool aggregation:** `MultiServerMCPClient.get_tools()` returns a flat list of tools from all configured servers; `bind_tools` doesn't care which server a tool came from.
