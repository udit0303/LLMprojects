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

### 3. [Streamlit Example](./StreamListExample1)
**Goal:** demonstrate rapid prototyping of AI interfaces.
- **Tech Stack:** Streamlit, Python.
- **Key Features:**
  - Simple, interactive UI components.
  - boilerplate patterns for LLM chat interfaces.

---

## 🚀 Getting Started

Each project is self-contained. Navigate to a folder to see its specific `README.md` for installation and setup instructions:

```bash
# Clone the repo
git clone https://github.com/yourusername/LLMProjects.git

# Navigate to a project
cd LLMProjects/MCP
# Follow instructions in MCP/README.md
```

## 📝 About
This repository explores the cutting edge of AI engineering—moving from simple prompt engineering to complex, multi-agent systems and grounded retrieval pipelines.
