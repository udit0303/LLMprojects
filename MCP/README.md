# MCP Wikipedia Agent

This project implements a Model Context Protocol (MCP) agent that can search Wikipedia. It consists of a server that exposes Wikipedia search tools and a client that uses a LangGraph-based agent to interact with these tools.

## Project Structure

- **`mcp_server.py`**: An MCP server that provides tools, prompts, and resources:
    - **Tools**:
        - Search Wikipedia for a topic (`fetch_wikipedia_info`).
        - List sections of a Wikipedia page (`list_wikipedia_sections`).
        - Get content of a specific section (`get_section_content`).
    - **Prompts**:
        - Identify important sections of a topic (`highlight_sections_prompt`).
    - **Resources**:
        - Access a list of suggested topics (`suggested_titles`).
- **`mcp_client.py`**: A client application that connects to the MCP server and runs a LangGraph agent. It uses OpenAI's GPT-4 to process user queries and call the appropriate Wikipedia tools. It also supports listing and executing MCP prompts and resources.
- **`suggested_titles.txt`**: A text file containing a list of suggested Wikipedia topics used by the `suggested_titles` resource.

## Prerequisites

- Python 3.10 or higher
- An OpenAI API Key

## Installation

1.  Clone the repository (if applicable) or navigate to the project directory.

2.  Install the required Python packages:

    ```bash
    pip install mcp langgraph langchain-openai langchain-core langchain-mcp-adapters wikipedia wikipedia_sections
    ```

## Configuration

> [!WARNING]
> **Security Notice**: The `mcp_client.py` file currently contains a hardcoded placeholder API key. **Do not use this key.**

Before running the client, you must configure your OpenAI API key.

1.  Open `mcp_client.py`.
2.  Locate the `ChatOpenAI` initialization (around line 34).
3.  Replace the `openai_api_key` value with your actual OpenAI API key, or better yet, use an environment variable:

    ```python
    # In mcp_client.py
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
python mcp_client.py
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
You can ask questions like:
- "Who is the CEO of Google?"
- "Tell me about the history of the Internet."
- "What are the sections in the Wikipedia page for Python (programming language)?"

### Prompt Mode
You can use MCP prompts defined on the server:

1.  **List available prompts**:
    ```text
    You: /prompts
    ```

2.  **Execute a prompt**:
    ```text
    You: /prompt highlight_sections_prompt "Artificial Intelligence"
    ```
    This will run the `highlight_sections_prompt` with the argument "Artificial Intelligence" and process the result through the agent.

### Resource Mode
You can access MCP resources defined on the server:

1.  **List available resources**:
    ```text
    You: /resources
    ```

2.  **Read a resource**:
    ```text
    You: /resource suggested_titles
    ```
    This will display the content of the `suggested_titles.txt` file served by the MCP server.

Type `exit`, `quit`, or `q` to stop the agent.
