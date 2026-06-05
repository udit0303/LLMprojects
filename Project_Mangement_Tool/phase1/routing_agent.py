# Test script for RoutingAgent

# TODO: 1 - Import the KnowledgeAugmentedPromptAgent and RoutingAgent
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'workflow_agents')))

from dotenv import load_dotenv
from base_agents import KnowledgeAugmentedPromptAgent, RoutingAgent

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# OpenAI key drives the router's embeddings; the Claude key drives the worker agents.
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

persona = "You are a college professor"

# TODO: 2 - Define the Texas Knowledge Augmented Prompt Agent
knowledge = "You know everything about Texas"
texas_agent = KnowledgeAugmentedPromptAgent(anthropic_api_key, persona, knowledge, model=model)

# TODO: 3 - Define the Europe Knowledge Augmented Prompt Agent
knowledge = "You know everything about Europe"
europe_agent = KnowledgeAugmentedPromptAgent(anthropic_api_key, persona, knowledge, model=model)

# TODO: 4 - Define the Math Knowledge Augmented Prompt Agent
persona = "You are a college math professor"
knowledge = "You know everything about math, you take prompts with numbers, extract math formulas, and show the answer without explanation"
math_agent = KnowledgeAugmentedPromptAgent(anthropic_api_key, persona, knowledge, model=model)

routing_agent = RoutingAgent(openai_api_key, [])
agents = [
    {
        "name": "texas agent",
        "description": "Answer a question about Texas",
        # TODO: 5 - Call the Texas Agent to respond to prompts
        "func": lambda x: texas_agent.respond(x),
    },
    {
        "name": "europe agent",
        "description": "Answer a question about Europe",
        # TODO: 6 - Define a function to call the Europe Agent
        "func": lambda x: europe_agent.respond(x),
    },
    {
        "name": "math agent",
        "description": "When a prompt contains numbers, respond with a math formula",
        # TODO: 7 - Define a function to call the Math Agent
        "func": lambda x: math_agent.respond(x),
    },
]

routing_agent.agents = agents

# TODO: 8 - Print the RoutingAgent responses to the prompts
prompts = [
    "Tell me about the history of Rome, Texas",
    "Tell me about the history of Rome, Italy",
    "One story takes 2 days, and there are 20 stories",
]

for prompt in prompts:
    print("\n" + "=" * 60)
    print(f"PROMPT: {prompt}")
    print("-" * 60)
    print(routing_agent.route(prompt))
