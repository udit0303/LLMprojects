# Test script for ActionPlanningAgent

# TODO: 1 - Import all required libraries, including the ActionPlanningAgent
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'workflow_agents')))

from dotenv import load_dotenv
from base_agents import ActionPlanningAgent

# TODO: 2 - Load environment variables (using Claude instead of OpenAI)
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

knowledge = """
# Fried Egg
1. Heat pan with oil or butter
2. Crack egg into pan
3. Cook until white is set (2-3 minutes)
4. Season with salt and pepper
5. Serve

# Scrambled Eggs
1. Crack eggs into a bowl
2. Beat eggs with a fork until mixed
3. Heat pan with butter or oil over medium heat
4. Pour egg mixture into pan
5. Stir gently as eggs cook
6. Remove from heat when eggs are just set but still moist
7. Season with salt and pepper
8. Serve immediately

# Boiled Eggs
1. Place eggs in a pot
2. Cover with cold water (about 1 inch above eggs)
3. Bring water to a boil
4. Remove from heat and cover pot
5. Let sit: 4-6 minutes for soft-boiled or 10-12 minutes for hard-boiled
6. Transfer eggs to ice water to stop cooking
7. Peel and serve
"""

# TODO: 3 - Instantiate the ActionPlanningAgent with the API key and knowledge
action_planning_agent = ActionPlanningAgent(anthropic_api_key, knowledge, model=model)

# TODO: 4 - Print the agent's extracted steps for the scrambled-eggs prompt
prompt = "One morning I wanted to have scrambled eggs"
steps = action_planning_agent.extract_steps_from_prompt(prompt)
print(f"Prompt: {prompt}\n")
print("Extracted steps:")
for step in steps:
    print(step)
