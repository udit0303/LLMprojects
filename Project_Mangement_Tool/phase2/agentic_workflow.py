# agentic_workflow.py

# TODO: 1 - Import the following agents: ActionPlanningAgent, KnowledgeAugmentedPromptAgent, EvaluationAgent, RoutingAgent from the workflow_agents.base_agents module
import os
import re
import sys
sys.path.append(os.path.dirname(__file__))
from dotenv import load_dotenv
from workflow_agents.base_agents import (
    ActionPlanningAgent,
    KnowledgeAugmentedPromptAgent,
    EvaluationAgent,
    RoutingAgent,
)

# TODO: 2 - Load the OpenAI key into a variable called openai_api_key
# Note: our Phase 1 agents run on Claude (anthropic_api_key); the RoutingAgent still
# needs an OpenAI key for embeddings, so we load both here.
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

# load the product spec
# TODO: 3 - Load the product spec document Product-Spec-Email-Router.txt into a variable called product_spec
with open(os.path.join(os.path.dirname(__file__), "Product-Spec-Email-Router.txt"), "r", encoding="utf-8") as f:
    product_spec = f.read()

# Instantiate all the agents

# Action Planning Agent
knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a "
    "persona, an action, and a desired outcome for each story. "
    "Each story represents a specific functionality of the product "
    "described in the specification. \n"
    "Features are defined by grouping related user stories. \n"
    "Tasks are defined for each story and represent the engineering "
    "work required to develop the product. \n"
    "A development Plan for a product contains all these components"
)
# TODO: 4 - Instantiate an action_planning_agent using the 'knowledge_action_planning'
action_planning_agent = ActionPlanningAgent(anthropic_api_key, knowledge_action_planning, model=model)

# Product Manager - Knowledge Augmented Prompt Agent
persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."
knowledge_product_manager = (
    "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
    "The sentences always start with: As a "
    "Write several stories for the product spec below, where the personas are the different users of the product. "
    # TODO: 5 - Complete this knowledge string by appending the product_spec loaded in TODO 3
    + product_spec
)
# TODO: 6 - Instantiate a product_manager_knowledge_agent using 'persona_product_manager' and the completed 'knowledge_product_manager'
product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    anthropic_api_key, persona_product_manager, knowledge_product_manager, model=model
)

# Product Manager - Evaluation Agent
# TODO: 7 - Define the persona and evaluation criteria for a Product Manager evaluation agent and instantiate it as product_manager_evaluation_agent. This agent will evaluate the product_manager_knowledge_agent.
# The evaluation_criteria should specify the expected structure for user stories (e.g., "As a [type of user], I want [an action or feature] so that [benefit/value].").
persona_product_manager_eval = "You are an evaluation agent that checks the answers of other worker agents"
evaluation_criteria_product_manager = (
    "The answer should be stories that follow the following structure: "
    "As a [type of user], I want [an action or feature] so that [benefit/value]."
)
product_manager_evaluation_agent = EvaluationAgent(
    anthropic_api_key,
    persona_product_manager_eval,
    evaluation_criteria_product_manager,
    worker_agent=product_manager_knowledge_agent,
    max_interactions=10,
    model=model,
)

# Program Manager - Knowledge Augmented Prompt Agent
persona_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."
knowledge_program_manager = "Features of a product are defined by organizing similar user stories into cohesive groups."
# Instantiate a program_manager_knowledge_agent using 'persona_program_manager' and 'knowledge_program_manager'
# (This is a necessary step before TODO 8. Students should add the instantiation code here.)
program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    anthropic_api_key, persona_program_manager, knowledge_program_manager, model=model
)

# Program Manager - Evaluation Agent
persona_program_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."

# TODO: 8 - Instantiate a program_manager_evaluation_agent using 'persona_program_manager_eval' and the evaluation criteria below.
#                      "The answer should be product features that follow the following structure: " \
#                      "Feature Name: A clear, concise title that identifies the capability\n" \
#                      "Description: A brief explanation of what the feature does and its purpose\n" \
#                      "Key Functionality: The specific capabilities or actions the feature provides\n" \
#                      "User Benefit: How this feature creates value for the user"
# For the 'agent_to_evaluate' parameter, refer to the provided solution code's pattern.
evaluation_criteria_program_manager = (
    "The answer should be product features that follow the following structure: "
    "Feature Name: A clear, concise title that identifies the capability\n"
    "Description: A brief explanation of what the feature does and its purpose\n"
    "Key Functionality: The specific capabilities or actions the feature provides\n"
    "User Benefit: How this feature creates value for the user"
)
program_manager_evaluation_agent = EvaluationAgent(
    anthropic_api_key,
    persona_program_manager_eval,
    evaluation_criteria_program_manager,
    worker_agent=program_manager_knowledge_agent,
    max_interactions=10,
    model=model,
)

# Development Engineer - Knowledge Augmented Prompt Agent
persona_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."
knowledge_dev_engineer = "Development tasks are defined by identifying what needs to be built to implement each user story."
# Instantiate a development_engineer_knowledge_agent using 'persona_dev_engineer' and 'knowledge_dev_engineer'
# (This is a necessary step before TODO 9. Students should add the instantiation code here.)
development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(
    anthropic_api_key, persona_dev_engineer, knowledge_dev_engineer, model=model
)

# Development Engineer - Evaluation Agent
persona_dev_engineer_eval = "You are an evaluation agent that checks the answers of other worker agents."
# TODO: 9 - Instantiate a development_engineer_evaluation_agent using 'persona_dev_engineer_eval' and the evaluation criteria below.
#                      "The answer should be tasks following this exact structure: " \
#                      "Task ID: A unique identifier for tracking purposes\n" \
#                      "Task Title: Brief description of the specific development work\n" \
#                      "Related User Story: Reference to the parent user story\n" \
#                      "Description: Detailed explanation of the technical work required\n" \
#                      "Acceptance Criteria: Specific requirements that must be met for completion\n" \
#                      "Estimated Effort: Time or complexity estimation\n" \
#                      "Dependencies: Any tasks that must be completed first"
# For the 'agent_to_evaluate' parameter, refer to the provided solution code's pattern.
evaluation_criteria_dev_engineer = (
    "The answer should be tasks following this exact structure: "
    "Task ID: A unique identifier for tracking purposes\n"
    "Task Title: Brief description of the specific development work\n"
    "Related User Story: Reference to the parent user story\n"
    "Description: Detailed explanation of the technical work required\n"
    "Acceptance Criteria: Specific requirements that must be met for completion\n"
    "Estimated Effort: Time or complexity estimation\n"
    "Dependencies: Any tasks that must be completed first"
)
development_engineer_evaluation_agent = EvaluationAgent(
    anthropic_api_key,
    persona_dev_engineer_eval,
    evaluation_criteria_dev_engineer,
    worker_agent=development_engineer_knowledge_agent,
    max_interactions=10,
    model=model,
)


# Job function persona support functions
# TODO: 11 - Define the support functions for the routes of the routing agent (e.g., product_manager_support_function, program_manager_support_function, development_engineer_support_function).
# Each support function should:
#   1. Take the input query (e.g., a step from the action plan).
#   2. Get a response from the respective Knowledge Augmented Prompt Agent.
#   3. Have the response evaluated by the corresponding Evaluation Agent.
#   4. Return the final validated response.
# Note: our EvaluationAgent.evaluate() already calls the worker agent's respond() internally
# and iterates until the criteria are met, returning the validated answer in 'final_response'.
#
# Step chaining: each routed step only receives its own step text, so without help the
# Program Manager / Development Engineer agents (which don't carry the product spec) would
# invent generic content. We accumulate the validated output of prior steps and feed it as
# context into each subsequent step, so PM user stories -> PgM features -> Dev tasks all
# build on the real Email Router work. The router still routes on the raw step text.
accumulated_context = ""

# The individual user stories produced by the Product Manager. We capture these so the
# Development Engineer can generate tasks one story at a time (see below) instead of
# emitting tasks for every story in a single response that would exceed the token limit.
user_stories = []


def _with_context(query):
    if not accumulated_context:
        return query
    return (
        "Work completed in previous steps of this project (use it as context, "
        "and stay consistent with it):\n"
        f"{accumulated_context}\n\n"
        "Now complete the following step:\n"
        f"{query}"
    )


def _extract_user_stories(text):
    """Pull individual 'As a ... so that ...' stories out of the PM's free-text output.

    Handles both standalone paragraphs and markdown table rows; deduplicates while
    preserving order. Returns [] if nothing matches so callers can fall back.
    """
    pattern = re.compile(r"(?i)^as an?\b.+?\bso that\b.+")
    stories = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*").strip()
        if "|" in line:  # markdown table row: find the cell holding the story
            for cell in line.split("|"):
                cell = cell.strip()
                if pattern.match(cell):
                    line = cell
                    break
        if pattern.match(line) and line not in stories:
            stories.append(line.rstrip("|").strip())
    return stories


def product_manager_support_function(query):
    global user_stories
    result = product_manager_evaluation_agent.evaluate(_with_context(query))["final_response"]
    # Remember the individual stories for the per-story task generation step.
    extracted = _extract_user_stories(result)
    if extracted:
        user_stories = extracted
    return result


def program_manager_support_function(query):
    result = program_manager_evaluation_agent.evaluate(_with_context(query))
    return result["final_response"]


def development_engineer_support_function(query):
    # Task generation for the whole product in one call would exceed the token limit, so
    # when we have the individual user stories we generate tasks one story at a time. Each
    # call is bounded (tasks for a single story), then we assemble. Falls back to a single
    # call if no stories were captured or this isn't a task-generation step.
    if user_stories and "task" in query.lower():
        print(f"[Dev Engineer] Generating tasks per user story ({len(user_stories)} stories)")
        sections = []
        for idx, story in enumerate(user_stories, 1):
            print(f"  - Story {idx}/{len(user_stories)}: {story[:80]}...")
            story_query = (
                "You are defining engineering tasks for the 'Email Router' product.\n"
                "Define the development tasks required to implement ONLY the following "
                f"single user story:\n{story}"
            )
            story_tasks = development_engineer_evaluation_agent.evaluate(story_query)["final_response"]
            sections.append(f"### Tasks for: {story}\n\n{story_tasks}")
        return "\n\n".join(sections)

    return development_engineer_evaluation_agent.evaluate(_with_context(query))["final_response"]


# Routing Agent
# TODO: 10 - Instantiate a routing_agent. You will need to define a list of agent dictionaries (routes) for Product Manager, Program Manager, and Development Engineer. Each dictionary should contain 'name', 'description', and 'func' (linking to a support function). Assign this list to the routing_agent's 'agents' attribute.
routing_agent = RoutingAgent(openai_api_key, [])
routing_agent.agents = [
    {
        "name": "Product Manager",
        "description": "Responsible for defining product personas and user stories only. Does not define features or tasks. Does not group stories.",
        "func": lambda x: product_manager_support_function(x),
    },
    {
        "name": "Program Manager",
        "description": "Responsible for defining product features by grouping related user stories. Does not define user stories or engineering tasks.",
        "func": lambda x: program_manager_support_function(x),
    },
    {
        "name": "Development Engineer",
        "description": "Responsible for defining detailed engineering development tasks to implement user stories. Does not define user stories or features.",
        "func": lambda x: development_engineer_support_function(x),
    },
]

# Run the workflow

print("\n*** Workflow execution started ***\n")
# Workflow Prompt
# ****
workflow_prompt = "What would the development tasks for this product be?"
# ****
print(f"Task to complete in this workflow, workflow prompt = {workflow_prompt}")

print("\nDefining workflow steps from the workflow prompt")
# TODO: 12 - Implement the workflow.
#   1. Use the 'action_planning_agent' to extract steps from the 'workflow_prompt'.
#   2. Initialize an empty list to store 'completed_steps'.
#   3. Loop through the extracted workflow steps:
#      a. For each step, use the 'routing_agent' to route the step to the appropriate support function.
#      b. Append the result to 'completed_steps'.
#      c. Print information about the step being executed and its result.
#   4. After the loop, print the final output of the workflow (the last completed step).
workflow_steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)
completed_steps = []

for i, step in enumerate(workflow_steps):
    print(f"\n================ Step {i + 1}/{len(workflow_steps)} ================")
    print(f"Step: {step}")

    is_final_step = (i == len(workflow_steps) - 1)
    if is_final_step and completed_steps:
        # The plan's final step is "compile all stories, features, and tasks into a
        # development plan". Routing it to an agent would make that agent re-expand every
        # prior artifact in a single response, which inevitably exceeds the token limit and
        # truncates. The validated artifacts already live in 'completed_steps', so we
        # assemble the plan deterministically here instead of re-generating it — this can
        # never truncate and costs no extra tokens.
        print("[Compile] Assembling development plan from prior validated artifacts (no re-generation)")
        result = "# Consolidated Development Plan\n\n" + "\n\n".join(
            f"## {prev_step.strip()}\n\n{prev_result}"
            for prev_step, prev_result in zip(workflow_steps[:-1], completed_steps)
        )
    else:
        # route() routes on the raw step text; the chosen support function injects
        # 'accumulated_context' (the validated output of all prior steps) into the prompt.
        result = routing_agent.route(step)

    completed_steps.append(result)
    # Feed this step's validated output forward into the next step's context.
    accumulated_context += f"\n\n## {step.strip()}\n{result}"
    print(f"\nResult of step {i + 1}:\n{result}")

print("\n*** Workflow execution completed ***\n")
print("Final output of the workflow:")
print(completed_steps[-1] if completed_steps else "(no steps were completed)")
