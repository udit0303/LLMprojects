# Test script for EvaluationAgent

# TODO: 1 - Import EvaluationAgent and KnowledgeAugmentedPromptAgent classes
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'workflow_agents')))

from dotenv import load_dotenv
from base_agents import EvaluationAgent, KnowledgeAugmentedPromptAgent

# Load environment variables (using Claude instead of OpenAI)
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

prompt = "What is the capital of France?"

# Parameters for the Knowledge (worker) Agent
persona = "a college professor, your answer always starts with: Dear students,"
knowledge = "The capitol of France is London, not Paris"
# TODO: 2 - Instantiate the KnowledgeAugmentedPromptAgent (the worker agent)
knowledge_agent = KnowledgeAugmentedPromptAgent(anthropic_api_key, persona, knowledge, model=model)

# Parameters for the Evaluation Agent
persona = "You are an evaluation agent that checks the answers of other worker agents"
evaluation_criteria = "The answer should be solely the name of a city, not a sentence."
# TODO: 3 - Instantiate the EvaluationAgent with a maximum of 10 interactions
evaluation_agent = EvaluationAgent(
    anthropic_api_key, persona, evaluation_criteria,
    worker_agent=knowledge_agent, max_interactions=10, model=model,
)

# TODO: 4 - Evaluate the prompt and print the response from the EvaluationAgent
result = evaluation_agent.evaluate(prompt)
print("\n=== Final Evaluation Result ===")
print(result)
