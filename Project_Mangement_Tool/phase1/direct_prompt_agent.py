# Test script for DirectPromptAgent

# TODO: 1 - Import the DirectPromptAgent class from the workflow_agents package
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'workflow_agents')))

from dotenv import load_dotenv
from base_agents import DirectPromptAgent

# TODO: 2 - Load the API key from the .env file (using Claude instead of OpenAI)
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

# TODO: 3 - Instantiate the DirectPromptAgent as direct_agent
direct_agent = DirectPromptAgent(anthropic_api_key, model=model)
prompt = "What is the Capital of France?"

# TODO: 4 - Use direct_agent's respond method, store the response in direct_agent_response
direct_agent_response = direct_agent.respond(prompt)

# TODO: 5 - Print the response from the agent
print(direct_agent_response)

# TODO: 6 - Knowledge source explanation:
# The DirectPromptAgent sends the prompt straight to Claude with no system prompt,
# no retrieved documents, and no tools. The answer therefore comes entirely from the
# model's own general (parametric) knowledge learned during training — not from any
# external or user-provided context.
print("\n[Knowledge source] This answer came purely from Claude's own general (parametric) "
      "knowledge — no system prompt, retrieved documents, or tools were used.")
