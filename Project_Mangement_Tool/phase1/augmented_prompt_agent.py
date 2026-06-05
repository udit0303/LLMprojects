# Test script for AugmentedPromptAgent

# TODO: 1 - Import the AugmentedPromptAgent class
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'workflow_agents')))

from dotenv import load_dotenv
from base_agents import AugmentedPromptAgent

# Load environment variables from .env file (using Claude instead of OpenAI)
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

prompt = "What is the capital of France?"
persona = "a college professor; your answers always start with: 'Dear students,'"

# TODO: 2 - Instantiate an object of AugmentedPromptAgent with the required parameters
augmented_agent = AugmentedPromptAgent(anthropic_api_key, persona, model=model)

# TODO: 3 - Send the 'prompt' to the agent and store the response
augmented_agent_response = augmented_agent.respond(prompt)

# Print the agent's response
print(augmented_agent_response)

# TODO: 4 - Explanation:
# - Knowledge used: The agent answered "Paris" from Claude's own general (parametric)
#   knowledge — no external documents or tools were provided.
# - Effect of the persona: The system prompt forced the model into the college-professor
#   persona, so the answer is reframed to open with "Dear students," and adopt an
#   instructional tone, instead of a plain factual reply.

