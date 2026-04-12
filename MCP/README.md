# MCP Wikipedia Agent

This project implements a **Model Context Protocol (MCP)** agent that can search and explore Wikipedia through natural conversation. It follows a client-server architecture where the server exposes Wikipedia capabilities as structured MCP primitives, and the client wraps them in a **LangGraph agentic loop** powered by GPT-4.

---

## Architecture

### High-Level Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                         USER (Terminal REPL)                      │
│   Free-text questions  |  /prompts  |  /resources  |  /prompt    │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                      mcp_client.py                                │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │                  LangGraph Agent Loop                      │   │
│  │                                                            │   │
│  │          ┌──────────────┐                                  │   │
│  │  START──►│  chat_node   │──► END (if no tool needed)       │   │
│  │          │  (GPT-4)     │                                  │   │
│  │          └──────┬───────┘                                  │   │
│  │                 │ (if tool call requested)                  │   │
│  │                 ▼                                           │   │
│  │          ┌──────────────┐                                  │   │
│  │          │  tool_node   │──► back to chat_node             │   │
│  │          │  (MCP tools) │                                  │   │
│  │          └──────────────┘                                  │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                   │
│  Also handles: /prompts, /resources, /prompt, /resource commands  │
└──────────────────────────────┬───────────────────────────────────┘
                               │ stdio (stdin/stdout pipe)
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                      mcp_server.py                                │
│                     (FastMCP "WikipediaSearch")                    │
│                                                                   │
│  Tools:      fetch_wikipedia_info, list_wikipedia_sections,       │
│              get_section_content                                  │
│  Prompts:    highlight_sections_prompt                            │
│  Resources:  file://suggested_titles                              │
└──────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                        Wikipedia API
```

### MCP Primitives

The server exposes three types of MCP primitives, each serving a different purpose:

| Primitive | Purpose | Invoked By |
|-----------|---------|------------|
| **Tools** | Callable functions the LLM can autonomously decide to invoke | The agent (via tool calls) |
| **Prompts** | Reusable prompt templates the user can trigger | The user (via `/prompt` command) |
| **Resources** | Read-only data endpoints (like REST GET) | The user (via `/resource` command) |

---

### Component Breakdown

#### `mcp_server.py` — The MCP Server

A **FastMCP server** named `"WikipediaSearch"` that runs over stdio transport (launched as a subprocess by the client).

**Tools:**

| Tool | Input | Output | Error Handling |
|------|-------|--------|----------------|
| `fetch_wikipedia_info(query)` | Search string | `{title, summary, url}` | Handles `DisambiguationError` (suggests alternatives) and `PageError` |
| `list_wikipedia_sections(topic)` | Topic name | `{sections: [...]}` | Generic exception catch |
| `get_section_content(topic, section_title)` | Topic + section name | `{content: "..."}` | Returns error if section not found |

**Prompt:**

```
highlight_sections_prompt(topic)
```
Generates a template asking the LLM to pick the 3–5 most important sections from a Wikipedia article and explain why each matters. When invoked via `/prompt`, the client feeds the filled template through the full agent, so the LLM can use tools (like `list_wikipedia_sections`) to actually fetch and analyze the sections.

**Resource:**

```
file://suggested_titles → suggested_titles.txt
```
Returns a list of pre-defined Wikipedia topics (coronavirus, python language, prompt engineering, node.js, photosynthesis, the solar system). Useful for bootstrapping a user's first interaction or as example queries.

---

#### `mcp_client.py` — The LangGraph Agent Client

Handles three responsibilities: **server lifecycle**, **agent orchestration**, and **user interaction**.

**1. Server Lifecycle**

```python
server_params = StdioServerParameters(command="python", args=["mcp_server.py"])
```
The client launches `mcp_server.py` as a subprocess and communicates over piped stdin/stdout. The `stdio_client` context manager handles process lifecycle automatically.

**2. Agent Orchestration (LangGraph)**

The agent is a stateful graph compiled with `MemorySaver` for cross-turn memory:

```
┌───────┐     tools_condition      ┌───────────┐
│       │ ──── "tools" ──────────► │           │
│ chat  │                          │   tool    │
│ node  │ ◄─────────────────────── │   node    │
│       │                          │           │
│       │ ──── "__end__" ──► END   │           │
└───────┘                          └───────────┘
```

- **`chat_node`**: Sends the full conversation history through a system prompt + GPT-4 (with tools bound). The LLM either responds directly or emits a tool call.
- **`tools_condition`**: LangGraph's built-in router that inspects the LLM response — if it contains a tool call, routes to `tool_node`; otherwise, ends.
- **`tool_node`**: Executes the requested MCP tool via the server and feeds the result back to `chat_node` for another reasoning pass.
- **`MemorySaver`**: All turns share `thread_id: "wiki-session"`, so the agent remembers prior context within a session.

**3. REPL Command Router**

| Command | Handler | Action |
|---------|---------|--------|
| Free text | `agent.ainvoke()` | Full agent loop with tool access |
| `/prompts` | `list_prompts()` | Lists all prompts + their argument schemas |
| `/prompt <name> "arg"` | `handle_prompt()` | Fetches template → fills args → runs through agent |
| `/resources` | `list_resources()` | Lists all server resources |
| `/resource <name>` | `handle_resource()` | Reads and displays resource content |
| `exit` / `quit` / `q` | — | Exits the REPL |

---

### End-to-End Flow Example

**User asks:** `"Tell me about photosynthesis"`

```
 1. User types: "Tell me about photosynthesis"
          │
 2. Client wraps it as HumanMessage, sends to LangGraph agent
          │
 3. chat_node: GPT-4 sees available tools, decides to call
    fetch_wikipedia_info("photosynthesis")
          │
 4. tools_condition: detects tool call → routes to tool_node
          │
 5. tool_node: sends request to MCP server over stdio pipe
          │
 6. MCP server: wikipedia.search("photosynthesis") → loads page
    → returns {title: "Photosynthesis", summary: "...", url: "..."}
          │
 7. tool_node: returns result as ToolMessage → back to chat_node
          │
 8. chat_node: GPT-4 reads the Wikipedia data,
    generates natural language response
          │
 9. tools_condition: no tool call → routes to END
          │
10. Client prints: "AI: Photosynthesis is the process by which
    green plants convert light energy into chemical energy..."
```

**User follows up:** `"What are the main sections?"`

```
 1. MemorySaver retains the previous turn's context
          │
 2. GPT-4 knows we were discussing photosynthesis,
    calls list_wikipedia_sections("Photosynthesis")
          │
 3. MCP server returns section list
          │
 4. GPT-4 formats and presents the sections
```

---

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| **stdio transport** | Simplest setup — no network config, no ports. Client manages server lifecycle automatically. |
| **Dynamic tool discovery** | `load_mcp_tools(session)` discovers tools at startup instead of hardcoding, so adding tools to the server requires zero client changes. |
| **LangGraph over plain LangChain** | Provides explicit control over the agent loop, conditional routing, and built-in memory. |
| **`MemorySaver` checkpointer** | Enables multi-turn conversations where follow-up questions understand prior context. |
| **Prompts as MCP primitives** | Separates prompt engineering from client logic — prompts can be updated on the server without touching the client. |
| **Resources as MCP primitives** | Provides a clean way to serve static data (suggested topics) without embedding it in tool logic. |

---

## Project Structure

```
MCP/
├── mcp_server.py           # FastMCP server: tools, prompts, resources
├── mcp_client.py           # LangGraph agent + interactive REPL
├── suggested_titles.txt    # Seed topics for the suggested_titles resource
└── README.md               # This file
```

---

## Prerequisites

- Python 3.10 or higher
- An OpenAI API Key

## Installation

1.  Clone the repository (if applicable) or navigate to the project directory.

2.  Install the required Python packages:

    ```bash
    pip install mcp langgraph langchain-openai langchain-core langchain-mcp-adapters wikipedia
    ```

## Configuration

> [!WARNING]
> **Security Notice**: The `mcp_client.py` file currently contains a hardcoded placeholder API key. **Do not use this key.**

Before running the client, you must configure your OpenAI API key.

1.  Open `mcp_client.py`.
2.  Locate the `ChatOpenAI` initialization (around line 30).
3.  Replace the `openai_api_key` value with your actual OpenAI API key, or better yet, use an environment variable:

    ```python
    # In single_server_mcp_client.py
    import os

    # ...

    llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=os.environ.get("OPENAI_API_KEY"))
    ```

    If using environment variables, make sure to set it in your terminal before running:
    
    ```bash
    export OPENAI_API_KEY="your-sk-..."
    ```

## Usage

To start the agent, run the client script. The client will automatically start the MCP server as a subprocess.

```bash
python single_server_mcp_client.py
```

Once started, you will see a prompt:

```text
Wikipedia MCP agent is ready.
Type a question or use the following templates:
  /prompts                - to list available prompts
  /prompt <name> "args"   - to run a specific prompt
  /resources              - to list available resources
  /resource <name>        - to run a specific resource

You:
```

### Chat Mode
Ask natural language questions — the agent will autonomously decide which Wikipedia tools to use:
- `"Who is the CEO of Google?"`
- `"Tell me about the history of the Internet."`
- `"What are the sections in the Wikipedia page for Python (programming language)?"`

### Prompt Mode
Use server-defined prompt templates for structured exploration:

1.  **List available prompts:**
    ```text
    You: /prompts
    ```

2.  **Execute a prompt:**
    ```text
    You: /prompt highlight_sections_prompt "Artificial Intelligence"
    ```
    This fetches the prompt template, fills in the topic, and runs the result through the full agent — so GPT-4 can use tools to actually analyze the article's sections.

### Resource Mode
Access static data served by the MCP server:

1.  **List available resources:**
    ```text
    You: /resources
    ```

2.  **Read a resource:**
    ```text
    You: /resource suggested_titles
    ```
    Displays the content of `suggested_titles.txt` — a curated list of topics to explore.

Type `exit`, `quit`, or `q` to stop the agent.
