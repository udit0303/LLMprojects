# RAG MCP

A retrieval-augmented generation (RAG) agent built with LangGraph and the Model Context Protocol (MCP). The client is a Claude-powered LangGraph agent; the server exposes ingestion and retrieval tools that read/write a persistent Chroma vector store.

## Architecture

| File | Role | Description |
|------|------|-------------|
| `Server.py` | MCP server (`RAGAssistant`) | Exposes `ingest_document` and `query_rag_store` tools over stdio. Uses OpenAI embeddings and a persistent Chroma store at `rag_chroma_db/`. |
| `Client.py` | LangGraph agent | Claude Sonnet 4.6 with tool-calling loop and `MemorySaver` for multi-turn context. Spawns `Server.py` as a subprocess automatically. |

---

## MCP Primitives

### `Server.py`

| Type | Name | Description |
|------|------|-------------|
| Tool | `ingest_document` | Loads a `.txt` file, splits it with `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap), embeds chunks via OpenAI `text-embedding-3-small`, and persists to Chroma. |
| Tool | `query_rag_store` | Runs a top-3 similarity search against the persisted vector store and returns the concatenated chunk text as context for the agent. |

---

## Setup

**Create a virtual environment and install dependencies:**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install mcp langgraph langchain-mcp-adapters langchain-anthropic \
            langchain-openai langchain-community langchain-chroma \
            langchain-text-splitters python-dotenv
```

**Configure API keys** — copy the example and fill in your keys:
```bash
cp .env.example .env
```

Edit `.env`:
```
ANTHROPIC_API_KEY=sk-ant-...    # from console.anthropic.com
OPENAI_API_KEY=sk-...           # from platform.openai.com (needed for embeddings)
```

> **Why OpenAI for embeddings?** Anthropic does not provide an embeddings API, so the chat LLM (Claude) and the embedding model (OpenAI) come from different providers.

---

## Running

```bash
python Client.py
```

The client spawns `Server.py` as a stdio subprocess automatically.

---

## Example Workflow

```
You: /absolute/path/to/employeeHandbook.txt
AI:  Successfully ingested the document from 'employeeHandbook.txt' into the vector store.

You: what is the vacation policy?
AI:  <answer grounded in the ingested document>

You: quit
```

**Accepted inputs for ingestion:** any conversational phrasing containing a file path works — the agent decides when to call `ingest_document`. Only plaintext `.txt` files are supported by the default loader.

---

## Files & Directories

```
RAG_MCP/
├── .env                 # your API keys (gitignored, you create it)
├── .env.example         # template
├── .gitignore
├── Client.py            # LangGraph agent (Claude)
├── Server.py            # FastMCP server
├── rag_chroma_db/       # auto-created: persistent Chroma vector store
└── venv/                # local virtualenv
```

---

## Notes

- **Persistence:** the Chroma store survives restarts. Delete `rag_chroma_db/` to reset the knowledge base.
- **Session memory:** the agent uses `MemorySaver` with a fixed `thread_id="rag-session"`, so follow-up questions in the same run retain conversational context.
- **Logs:** HTTP request logs from `httpx`, `openai`, and `anthropic` clients are silenced to `WARNING` to keep REPL output clean.
