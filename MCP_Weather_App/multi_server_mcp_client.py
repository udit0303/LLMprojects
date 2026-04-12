import asyncio
import shlex
from mcp import StdioServerParameters
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition, ToolNode
from typing import Annotated, List
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Import the MultiServerMCPClient
from langchain_mcp_adapters.client import MultiServerMCPClient

# --- Multi-server configuration dictionary ---
# This dictionary defines all the servers the client will connect to.
server_configs = {
    "weather": {
        "command": "python",
        "args": ["weather_server.py"],  # The weather server
        "transport": "stdio",
    },
    "tasks": {
        "command": "python",
        "args": ["task_server.py"],  # The new task management server
        "transport": "stdio",
    }
}


# LangGraph state definition (remains the same)
class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]


# --- 'create_graph' now accepts the list of tools directly ---
def create_graph(tools: list):
    # LLM configuration (remains the same)
    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0, api_key="<<INSERT_KEY>>")
    llm_with_tools = llm.bind_tools(tools)

    # --- Updated system prompt to reflect new capabilities ---
    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. You have access to tools for checking the weather and managing a to-do list. Use the tools when necessary based on the user's request."),
        MessagesPlaceholder("messages")
    ])

    chat_llm = prompt_template | llm_with_tools

    # Define chat node (remains the same)
    def chat_node(state: State) -> State:
        response = chat_llm.invoke({"messages": state["messages"]})
        return {"messages": [response]}

    # Build LangGraph with tool routing (remains the same)
    graph = StateGraph(State)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tool_node", ToolNode(tools=tools))
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition, {
        "tools": "tool_node",
        "__end__": END
    })
    graph.add_edge("tool_node", "chat_node")

    return graph.compile(checkpointer=MemorySaver())


# Iterates through all configured servers, lists their available prompts, and prints them in a user-friendly format

async def list_all_prompts(client: MultiServerMCPClient, server_configs: dict):
    print("\nAvailable Prompts from all servers:")
    print("-----------------------------------")

    any_prompts_found = False

    # Iterate through the names of the servers defined in your server_configs
    for server_name in server_configs.keys():
        try:
            # Opening a session for a specific server to interact with its prompts
            async with client.session(server_name) as session:
                prompt_response = await session.list_prompts()

                if prompt_response and prompt_response.prompts:
                    any_prompts_found = True
                    # Print a header for the server to group the prompts
                    print(f"\n--- Server: '{server_name}' ---")
                    for p in prompt_response.prompts:
                        print(f"  Prompt: {p.name}")
                        if p.arguments:
                            arg_list = [f"{arg.name}" for arg in p.arguments]
                            print(f"    Arguments: {', '.join(arg_list)}")
                        else:
                            print("    Arguments: None")
        except Exception as e:
            print(f"\nCould not fetch prompts from server '{server_name}': {e}")

    print("\nUse: /prompt <server_name> <prompt_name> \"arg1\" \"arg2\" ...")
    print("-----------------------------------")
    if not any_prompts_found:
        print("\nNo prompts were found on any connected servers.")


async def list_all_resources(client: MultiServerMCPClient, server_configs: dict):
    print("\nAvailable Resources from all servers:")
    print("--------------------------------------")

    any_found = False

    for server_name in server_configs.keys():
        try:
            async with client.session(server_name) as session:
                resource_response = await session.list_resources()

                if resource_response and resource_response.resources:
                    any_found = True
                    print(f"\n--- Server: '{server_name}' ---")
                    for r in resource_response.resources:
                        print(f"  Resource: {r.uri}  ({r.name})")
        except Exception as e:
            print(f"\nCould not fetch resources from server '{server_name}': {e}")

    print("\nUse: /resource <server_name> <resource_uri>")
    print("--------------------------------------")
    if not any_found:
        print("\nNo resources were found on any connected servers.")


async def handle_resource(client: MultiServerMCPClient, command: str) -> str | None:
    parts = command.strip().split(None, 2)

    if len(parts) < 3:
        print("\nUsage: /resource <server_name> <resource_uri>")
        return None

    server_name = parts[1]
    resource_uri = parts[2]

    try:
        print(f"\n--- Fetching resource '{resource_uri}' from server '{server_name}'... ---")
        async with client.session(server_name) as session:
            result = await session.read_resource(resource_uri)

        contents = []
        for item in result.contents:
            if hasattr(item, "text"):
                contents.append(item.text)

        if not contents:
            print("Resource returned no content.")
            return None

        print("--- Resource loaded successfully. ---")
        return "\n".join(contents)

    except Exception as e:
        print(f"\nError fetching resource: {e}")
        return None


# Parses a user command to invoke a specific prompt on a specific server,
# then executes the resulting workflow with the agent
async def handle_prompt_invocation(client: MultiServerMCPClient, command: str) -> str | None:
    try:
        # Use shlex to correctly parse arguments, even with spaces
        parts = shlex.split(command.strip())

        # Command should be: /prompt <server_name> <prompt_name> [args...]
        if len(parts) < 3:
            print("\nUsage: /prompt <server_name> <prompt_name> \"arg1\" \"arg2\" ...")
            return None

        server_name = parts[1]
        prompt_name = parts[2]
        user_args = parts[3:]

        # --- Validate the prompt and arguments against the specific server ---
        prompt_def = None
        async with client.session(server_name) as session:
            all_prompts_resp = await session.list_prompts()
            if all_prompts_resp and all_prompts_resp.prompts:
                # Find the matching prompt definition
                prompt_def = next((p for p in all_prompts_resp.prompts if p.name == prompt_name), None)

        if not prompt_def:
            print(f"\nError: Prompt '{prompt_name}' not found on server '{server_name}'.")
            return None

        # Check if the number of user-provided arguments matches what the prompt expects
        if len(user_args) != len(prompt_def.arguments):
            expected_args = [arg.name for arg in prompt_def.arguments]
            print(f"\nError: Invalid number of arguments for prompt '{prompt_name}'.")
            print(f"Expected {len(expected_args)} arguments: {', '.join(expected_args)}")
            return

        # --- Coerce args to expected types and build arg dict ---
        import re
        def coerce_arg(arg_def, value: str):
            # If the argument name suggests an integer type, extract digits
            if arg_def.description and "int" in arg_def.description.lower():
                match = re.search(r'\d+', value)
                if match:
                    return str(int(match.group()))
            # Fallback: try extracting leading digits for args with "days", "count", "num" in name
            if any(hint in arg_def.name.lower() for hint in ("days", "count", "num", "duration", "hours", "minutes")):
                match = re.search(r'\d+', value)
                if match:
                    return str(int(match.group()))
            return value

        arg_dict = {arg.name: coerce_arg(arg, val) for arg, val in zip(prompt_def.arguments, user_args)}

        # Use the client's get_prompt method, specifying server, prompt, and args
        prompt_messages = await client.get_prompt(
            server_name=server_name,
            prompt_name=prompt_name,
            arguments=arg_dict
        )

        # The prompt text is the content of the first message returned
        prompt_text = prompt_messages[0].content

        print("\n--- Prompt loaded successfully. Preparing to execute... ---")

        # Return the fetched text instead of calling the agent
        return prompt_text

    except Exception as e:
        print(f"\nAn error occurred during prompt invocation: {e}")


# --- Main function ---
async def main():
    # Instantiate the client which will manage the server subprocesses internally
    client = MultiServerMCPClient(server_configs)

    # Get a single, unified list of tools from all connected servers
    all_tools = await client.get_tools()

    # Create the LangGraph agent with the aggregated list of tools
    agent = create_graph(all_tools)

    print("MCP Agent is ready (connected to Weather and Task servers).")
    # Update the instructions for the user
    print("Type a question, or use one of the following commands:")
    print("  /prompts                                              - to list available prompts")
    print("  /prompt <server_name> <prompt_name> \"args\"          - to run a specific prompt")
    print("  /resources                                            - to list available resources")
    print("  /resource <server_name> <resource_uri>               - to fetch a specific resource")

    pending_resource_content = None

    while True:
        # This variable will hold the final message to be sent to the agent
        message_to_agent = ""

        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit", "q"}:
            break

        # --- Command Handling Logic ---
        if user_input.lower() == "/prompts":
            await list_all_prompts(client, server_configs)
            continue  # Command is done, loop back for next input

        elif user_input.startswith("/prompt"):
            prompt_text = await handle_prompt_invocation(client, user_input)
            if prompt_text:
                message_to_agent = prompt_text
            else:
                continue

        elif user_input.lower() == "/resources":
            await list_all_resources(client, server_configs)
            continue

        elif user_input.startswith("/resource"):
            resource_content = await handle_resource(client, user_input)
            if resource_content:
                follow_up = input("What would you like to do with this resource? ").strip()
                if not follow_up:
                    pending_resource_content = resource_content
                    print("Resource content loaded. You can reference it in your next message.")
                    continue
                message_to_agent = f"Here is the resource content:\n{resource_content}\n\nUser request: {follow_up}"
            else:
                continue

        else:
            # For a normal chat message, attach any pending resource content
            if pending_resource_content:
                message_to_agent = f"Here is the resource content:\n{pending_resource_content}\n\nUser request: {user_input}"
                pending_resource_content = None
            else:
                message_to_agent = user_input

        # --- Final agent invocation ---

        if message_to_agent:
            try:
                response = await agent.ainvoke(
                    {"messages": [("user", message_to_agent)]},
                    config={"configurable": {"thread_id": "multi-server-session"}}
                )
                print("AI:", response["messages"][-1].content)
            except Exception as e:
                print("Error:", e)


if __name__ == "__main__":
    asyncio.run(main())