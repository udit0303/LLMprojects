# LLM Projects Portfolio

Welcome to my collection of Large Language Model (LLM) projects. This repository showcases different ways to build practical applications with AI, from RAG systems to autonomous agents.

## 📂 Projects

### 1. [MedicalBot (RAG Assistant)](./MedicalBot)
**Goal:** Answer patient questions about food-drug interactions using authoritative medical sources.
- **Tech Stack:** Python, LangChain, OpenAI, ChromaDB, Streamlit.
- **Key Features:**
  - Automated scraping of MedlinePlus and NHS UK.
  - Drug name normalization via RxNav API.
  - RAG pipeline for grounded, citation-backed answers.

### 2. [MCP Wikipedia Agent](./MCP)
**Goal:** Build a "universal" autonomous research agent using the Model Context Protocol (MCP).
- **Tech Stack:** Python, LangGraph, FastMCP, GPT-4.
- **Key Features:**
  - **Server:** Exposes Wikipedia search tools via standardized MCP.
  - **Client:** A LangGraph agent that autonomously reasons and calls tools.
  - **Decoupled Architecture:** Client logic is separated from tool implementation.

### 3. [MCP Weather App](./MCP_Weather_App)
**Goal:** Build a multi-server MCP agentic system with weather and task management capabilities.
- **Tech Stack:** Python, LangGraph, FastMCP, Claude (Anthropic), OpenWeatherMap API.
- **Key Features:**
  - **Two MCP Servers:** A weather server and a task management server, each exposing tools, prompts, and resources.
  - **Single-server client:** LangGraph agent (Claude) connected to the weather server only.
  - **Multi-server client:** LangGraph agent (Claude) connected to both servers via `MultiServerMCPClient`.
  - **All three MCP primitives:** Tools (get weather, manage tasks), Prompts (compare cities, plan trips), Resources (delivery log, meeting notes).
  - **Persistent tasks:** Task list stored in a plain text file across sessions.

### 4. [RAG MCP](./RAG_MCP)
**Goal:** Expose retrieval-augmented generation (RAG) as a reusable MCP capability — let a Claude-powered agent ingest documents and answer questions grounded in them.
- **Tech Stack:** Python, LangGraph, FastMCP, Claude (Anthropic), OpenAI Embeddings, Chroma.
- **Key Features:**
  - **MCP server:** Two tools — `ingest_document` (chunks, embeds, persists) and `query_rag_store` (top-K similarity search).
  - **LangGraph agent (Claude Sonnet 4.6):** Same chat-node → tool-node loop as the weather agent — MCP decouples the agent from capability.
  - **Cross-provider design:** Claude for chat, OpenAI `text-embedding-3-small` for vectors (Anthropic has no embeddings API).
  - **Persistent Chroma store:** Ingested knowledge survives restarts.
  - **`.env` driven config:** `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` loaded via `python-dotenv`.

### 5. [Image Research Assistant](./ImageAssistant)
**Goal:** Build a multi-server MCP agent with a web UI that analyzes uploaded images and chains the result into Wikipedia research.
- **Tech Stack:** Python, LangGraph, FastMCP, Claude Sonnet 4.6 + Haiku 4.5 (Anthropic), Gradio, Wikipedia.
- **Key Features:**
  - **Two MCP servers:** `visual_analysis_server.py` (vision via Claude Haiku 4.5) and `wikipedia_server.py` (Wikipedia search).
  - **Tool chaining:** The agent describes an image, then uses the description as a Wikipedia query — all in one user turn.
  - **Two-tier model use:** Sonnet 4.6 for reasoning/tool selection, Haiku 4.5 for vision — cheap and fast where it matters.
  - **Gradio web UI:** Image upload + chat in a browser, with a Quit button that shuts the process down cleanly.
  - **Server-side image processing:** Base64 stays inside the MCP server, never crossing the LLM context — keeps token usage lean.

### 6. [Streamlit Example](./StreamListExample1)
**Goal:** demonstrate rapid prototyping of AI interfaces.
- **Tech Stack:** Streamlit, Python.
- **Key Features:**
  - Simple, interactive UI components.
  - boilerplate patterns for LLM chat interfaces.

### 7. [Project Management Tool (Agentic Workflow)](./Project_Mangement_Tool)
**Goal:** Build a reusable library of AI agents, then compose them into a multi-agent workflow that turns a product spec into a structured development plan (personas → user stories → features → engineering tasks).
- **Tech Stack:** Python, Claude (Anthropic), OpenAI Embeddings.
- **Key Features:**
  - **Phase 1 — agent library:** Seven reusable agent classes (DirectPrompt, AugmentedPrompt, KnowledgeAugmentedPrompt, RAGKnowledgePrompt, Evaluation, Routing, ActionPlanning), each with a standalone test script.
  - **Phase 2 — agentic workflow:** Action planning → embedding-based routing → specialist agents (Product Manager / Program Manager / Development Engineer), each validated by an `EvaluationAgent` loop.
  - **Context chaining:** Each step builds on prior validated output, keeping the plan grounded in the real product spec.
  - **Bounded generation:** Deterministic final compile (assembled in code) and per-user-story task generation so no single model call ever truncates.
  - **Cross-provider design:** Claude for all reasoning, OpenAI only for embeddings (routing + RAG).

---

## 🚀 Getting Started

Each project is self-contained. Navigate to a folder to see its specific `README.md` for installation and setup instructions:

```bash
# Clone the repo
git clone https://github.com/udit0303/LLMprojects.git

# Navigate to a project
cd LLMProjects/MCP
# Follow instructions in MCP/README.md
```

## 📝 About
This repository explores the cutting edge of AI engineering—moving from simple prompt engineering to complex, multi-agent systems and grounded retrieval pipelines.
