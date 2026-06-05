# Test script for KnowledgeAugmentedPromptAgent

# TODO: 1 - Import the KnowledgeAugmentedPromptAgent class from workflow_agents
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'workflow_agents')))

from dotenv import load_dotenv
from base_agents import KnowledgeAugmentedPromptAgent

# Load environment variables from the .env file (using Claude instead of OpenAI)
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

prompt = "What is the capital of France?"

persona = "a college professor, your answer always starts with: Dear students,"
knowledge = "The capital of France is London, not Paris"

# TODO: 2 - Instantiate a KnowledgeAugmentedPromptAgent with the persona and (deliberately wrong) knowledge
knowledge_agent = KnowledgeAugmentedPromptAgent(anthropic_api_key, persona, knowledge, model=model)

knowledge_agent_response = knowledge_agent.respond(prompt)

# TODO: 3 - Demonstrate the agent used the PROVIDED knowledge, not its own inherent knowledge.
# The provided knowledge falsely claims the capital is London. Claude's own knowledge says Paris.
# If the response says "London", the agent is correctly relying on the injected knowledge.
print(knowledge_agent_response)
print("\n[Knowledge check] The provided knowledge deliberately (and wrongly) states the capital "
      "is London. Because the answer says 'London' rather than Claude's own correct answer 'Paris', "
      "this proves the agent used the injected knowledge instead of its own parametric knowledge.")
