from anthropic import Anthropic  # chat-based agents run on Claude
from openai import OpenAI  # used only for embeddings (RAGKnowledgePromptAgent + RoutingAgent)
import numpy as np
import pandas as pd
import re
import csv
import uuid
from datetime import datetime

# DirectPromptAgent class definition
class DirectPromptAgent:
    """Relays a user prompt straight to Claude with no system prompt, context, or tools.
    Responses come purely from the model's own general (parametric) knowledge."""

    def __init__(self, anthropic_api_key, model="claude-sonnet-4-6"):
        # Store the Anthropic API key used to authenticate requests.
        self.anthropic_api_key = anthropic_api_key
        self.model = model

    def respond(self, prompt):
        # Generate a response using the Anthropic Messages API.
        client = Anthropic(api_key=self.anthropic_api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=16384,
            temperature=0,
            messages=[
                # The user's prompt is sent directly — no system prompt.
                {"role": "user", "content": prompt}
            ],
        )
        # Return only the text of the response, not the full message object.
        return response.content[0].text

# AugmentedPromptAgent class definition
class AugmentedPromptAgent:
    """Responds in a defined persona. The persona is injected as a Claude system prompt,
    and the agent is told to forget any prior conversational context."""

    def __init__(self, anthropic_api_key, persona, model="claude-sonnet-4-6"):
        # Store the persona and credentials for this agent.
        self.persona = persona
        self.anthropic_api_key = anthropic_api_key
        self.model = model

    def respond(self, input_text):
        # Generate a response using the Anthropic Messages API.
        client = Anthropic(api_key=self.anthropic_api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=16384,
            temperature=0,
            # System prompt establishes the persona and explicitly clears prior context.
            system=f"You are {self.persona}. Forget all previous context.",
            messages=[
                {"role": "user", "content": input_text}
            ],
        )
        # Return only the text of the response, not the full message object.
        return response.content[0].text

# KnowledgeAugmentedPromptAgent class definition
class KnowledgeAugmentedPromptAgent:
    """Answers using a defined persona AND a specific block of provided knowledge,
    instructed to rely only on that knowledge rather than the model's own."""

    def __init__(self, anthropic_api_key, persona, knowledge, model="claude-sonnet-4-6"):
        self.persona = persona
        self.knowledge = knowledge
        self.anthropic_api_key = anthropic_api_key
        self.model = model

    def respond(self, input_text):
        client = Anthropic(api_key=self.anthropic_api_key)
        # System prompt: persona + the provided knowledge + an instruction to use only that knowledge.
        system_prompt = (
            f"You are {self.persona} knowledge-based assistant. Forget all previous context.\n"
            f"Use only the following knowledge to answer, do not use your own knowledge: {self.knowledge}\n"
            f"Answer the prompt based on this knowledge, not your own."
        )
        response = client.messages.create(
            model=self.model,
            max_tokens=16384,
            temperature=0,
            system=system_prompt,
            messages=[
                {"role": "user", "content": input_text}
            ],
        )
        return response.content[0].text


# RAGKnowledgePromptAgent class definition
class RAGKnowledgePromptAgent:
    """
    An agent that uses Retrieval-Augmented Generation (RAG) to find knowledge from a large corpus
    and leverages embeddings to respond to prompts based solely on retrieved information.
    """

    def __init__(self, openai_api_key, persona, anthropic_api_key=None,
                 model="claude-sonnet-4-6", chunk_size=2000, chunk_overlap=100):
        """
        Initializes the RAGKnowledgePromptAgent with API credentials and configuration settings.

        Parameters:
        openai_api_key (str): API key for OpenAI — used ONLY for embeddings (Claude has none).
        persona (str): Persona description for the agent.
        anthropic_api_key (str): API key for Anthropic — used for the Claude chat completion.
        model (str): Claude model used to generate the final answer.
        chunk_size (int): The size of text chunks for embedding. Defaults to 2000.
        chunk_overlap (int): Overlap between consecutive chunks. Defaults to 100.
        """
        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key
        self.model = model
        self.unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"

    def get_embedding(self, text):
        """
        Fetches the embedding vector for given text using OpenAI's embedding API.

        Parameters:
        text (str): Text to embed.

        Returns:
        list: The embedding vector.
        """
        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding

    def calculate_similarity(self, vector_one, vector_two):
        """
        Calculates cosine similarity between two vectors.

        Parameters:
        vector_one (list): First embedding vector.
        vector_two (list): Second embedding vector.

        Returns:
        float: Cosine similarity between vectors.
        """
        vec1, vec2 = np.array(vector_one), np.array(vector_two)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def chunk_text(self, text):
        """
        Splits text into manageable chunks, attempting natural breaks.

        Parameters:
        text (str): Text to split into chunks.

        Returns:
        list: List of dictionaries containing chunk metadata.
        """
        separator = "\n"
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) <= self.chunk_size:
            # Single-chunk path. Build the chunk list here (no early return) so the
            # CSV write below still runs — otherwise calculate_embeddings() would
            # later read a chunks-*.csv that was never written (FileNotFoundError).
            chunks = [{"chunk_id": 0, "text": text, "chunk_size": len(text)}]
        else:
            chunks, start, chunk_id = [], 0, 0

            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                if separator in text[start:end]:
                    end = start + text[start:end].rindex(separator) + len(separator)

                chunks.append({
                    "chunk_id": chunk_id,
                    "text": text[start:end],
                    "chunk_size": end - start,
                    "start_char": start,
                    "end_char": end
                })

                # Stop once the window reaches the end of the text. Without this, the final
                # window pins `end` to len(text) and `start = end - chunk_overlap` becomes a
                # fixed point, spinning forever (the OS then SIGKILLs the process, exit 137).
                if end >= len(text):
                    break

                start = end - self.chunk_overlap
                chunk_id += 1

        with open(f"chunks-{self.unique_filename}", 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
            writer.writeheader()
            for chunk in chunks:
                writer.writerow({k: chunk[k] for k in ["text", "chunk_size"]})

        return chunks

    def calculate_embeddings(self):
        """
        Calculates embeddings for each chunk and stores them in a CSV file.

        Returns:
        DataFrame: DataFrame containing text chunks and their embeddings.
        """
        df = pd.read_csv(f"chunks-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['text'].apply(self.get_embedding)
        df.to_csv(f"embeddings-{self.unique_filename}", encoding='utf-8', index=False)
        return df

    def find_prompt_in_knowledge(self, prompt):
        """
        Finds and responds to a prompt based on similarity with embedded knowledge.

        Parameters:
        prompt (str): User input prompt.

        Returns:
        str: Response derived from the most similar chunk in knowledge.
        """
        prompt_embedding = self.get_embedding(prompt)
        df = pd.read_csv(f"embeddings-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['embeddings'].apply(lambda x: np.array(eval(x)))
        df['similarity'] = df['embeddings'].apply(lambda emb: self.calculate_similarity(prompt_embedding, emb))

        best_chunk = df.loc[df['similarity'].idxmax(), 'text']

        # Embeddings come from OpenAI above; the final answer is generated by Claude.
        client = Anthropic(api_key=self.anthropic_api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=16384,
            temperature=0,
            system=f"You are {self.persona}, a knowledge-based assistant. Forget previous context.",
            messages=[
                {"role": "user", "content": f"Answer based only on this information: {best_chunk}. Prompt: {prompt}"}
            ],
        )

        return response.content[0].text


class EvaluationAgent:
    """Evaluates a worker agent's response against a set of criteria, iterating up to
    max_interactions and feeding correction instructions back until the criteria are met."""

    def __init__(self, anthropic_api_key, persona, evaluation_criteria, worker_agent,
                 max_interactions, model="claude-sonnet-4-6"):
        # Initialize the EvaluationAgent with given attributes.
        self.anthropic_api_key = anthropic_api_key
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.worker_agent = worker_agent
        self.max_interactions = max_interactions
        self.model = model

    def evaluate(self, initial_prompt):
        # This method manages interactions between agents to achieve a solution.
        client = Anthropic(api_key=self.anthropic_api_key)
        prompt_to_evaluate = initial_prompt

        for i in range(self.max_interactions):
            print(f"\n--- Interaction {i+1} ---")

            print(" Step 1: Worker agent generates a response to the prompt")
            print(f"Prompt:\n{prompt_to_evaluate}")
            response_from_worker = self.worker_agent.respond(prompt_to_evaluate)
            print(f"Worker Agent Response:\n{response_from_worker}")

            print(" Step 2: Evaluator agent judges the response")
            eval_prompt = (
                f"Does the following answer: {response_from_worker}\n"
                f"Meet this criteria: {self.evaluation_criteria}\n"
                f"Respond Yes or No, and the reason why it does or doesn't meet the criteria."
            )
            # Evaluator runs on Claude; the evaluator persona is supplied as the system prompt.
            response = client.messages.create(
                model=self.model,
                max_tokens=16384,
                temperature=0,
                system=self.persona,
                messages=[{"role": "user", "content": eval_prompt}],
            )
            evaluation = response.content[0].text.strip()
            print(f"Evaluator Agent Evaluation:\n{evaluation}")

            print(" Step 3: Check if evaluation is positive")
            # Claude often wraps the verdict in markdown (e.g. "**Yes.**"), so strip any
            # leading non-letter characters before checking rather than relying on plain text.
            verdict = evaluation.lower().lstrip("*#>-_ \t\n.")
            if verdict.startswith("yes"):
                print("✅ Final solution accepted.")
                break
            else:
                print(" Step 4: Generate instructions to correct the response")
                instruction_prompt = (
                    f"Provide instructions to fix an answer based on these reasons why it is incorrect: {evaluation}"
                )
                response = client.messages.create(
                    model=self.model,
                    max_tokens=16384,
                    temperature=0,
                    messages=[{"role": "user", "content": instruction_prompt}],
                )
                instructions = response.content[0].text.strip()
                print(f"Instructions to fix:\n{instructions}")

                print(" Step 5: Send feedback to worker agent for refinement")
                prompt_to_evaluate = (
                    f"The original prompt was: {initial_prompt}\n"
                    f"The response to that prompt was: {response_from_worker}\n"
                    f"It has been evaluated as incorrect.\n"
                    f"Make only these corrections, do not alter content validity: {instructions}"
                )
        return {
            "final_response": response_from_worker,
            "evaluation": evaluation,
            "iterations": i + 1,
        }

class RoutingAgent():
    """Routes a user prompt to the most relevant agent by comparing the prompt's embedding
    against each agent's description embedding (highest cosine similarity wins).
    Embeddings use OpenAI (Claude has none); the routed agents themselves run on Claude."""

    def __init__(self, openai_api_key, agents):
        # Initialize the agent with given attributes
        self.openai_api_key = openai_api_key
        self.agents = agents

    def get_embedding(self, text):
        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )
        # Extract and return the embedding vector from the response
        embedding = response.data[0].embedding
        return embedding

    def route(self, user_input):
        # Compute the embedding of the user input prompt
        input_emb = self.get_embedding(user_input)
        best_agent = None
        best_score = -1

        for agent in self.agents:
            # Compute the embedding of the agent description
            agent_emb = self.get_embedding(agent["description"])
            if agent_emb is None:
                continue

            similarity = np.dot(input_emb, agent_emb) / (np.linalg.norm(input_emb) * np.linalg.norm(agent_emb))
            print(similarity)

            # Select the agent whose description is most similar to the prompt
            if similarity > best_score:
                best_score = similarity
                best_agent = agent

        if best_agent is None:
            return "Sorry, no suitable agent could be selected."

        print(f"[Router] Best agent: {best_agent['name']} (score={best_score:.3f})")
        return best_agent["func"](user_input)

class ActionPlanningAgent:
    """Uses provided knowledge to extract the ordered steps needed to complete the
    task described in a user's prompt, returned as a clean list of action strings."""

    def __init__(self, anthropic_api_key, knowledge, model="claude-sonnet-4-6"):
        # Initialize the agent attributes.
        self.anthropic_api_key = anthropic_api_key
        self.knowledge = knowledge
        self.model = model

    def extract_steps_from_prompt(self, prompt):
        # Instantiate the Anthropic client and ask Claude to extract the steps.
        client = Anthropic(api_key=self.anthropic_api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=16384,
            temperature=0,
            system=(
                "You are an action planning agent. Using your knowledge, you extract from the user "
                "prompt the steps requested to complete the action the user is asking for. You return "
                "the steps as a list. Only return the steps in your knowledge. Forget any previous "
                "context. Respond with ONLY the list of steps, one concise step per line, with no "
                "preamble, no commentary, and never ask for clarification. You do not need any "
                "specific product, document, or details to answer — output only the high-level "
                "sequence of planning steps described in your knowledge. "
                f"This is your knowledge: {self.knowledge}"
            ),
            messages=[{"role": "user", "content": prompt}],
        )
        # Extract the response text from the Anthropic API response.
        response_text = response.content[0].text

        # Clean and format the extracted steps by removing empty/blank lines.
        steps = [step for step in response_text.split("\n") if step.strip()]

        return steps
