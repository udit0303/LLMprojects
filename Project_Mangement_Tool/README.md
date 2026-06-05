# AI-Powered Agentic Workflow for Project Management

A two-phase project that builds a reusable library of AI agents (Phase 1) and uses them
to run a multi-agent project-management workflow (Phase 2). Given a product specification,
the workflow plans the work, routes each step to the right specialist agent, and produces a
structured development plan (personas → user stories → features → engineering tasks).

The agents run on **Anthropic Claude** for all chat/reasoning, and use **OpenAI embeddings**
only where vector similarity is required (Claude has no embeddings API).

---

## Repository layout

```
Project_Mangement_Tool/
├── phase1/                          # Agent library + standalone tests
│   ├── workflow_agents/
│   │   ├── __init__.py
│   │   └── base_agents.py           # The 7 agent classes (core deliverable)
│   ├── direct_prompt_agent.py       # One test script per agent
│   ├── augmented_prompt_agent.py
│   ├── knowledge_augmented_prompt_agent.py
│   ├── rag_knowledge_prompt_agent.py
│   ├── evaluation_agent.py
│   ├── routing_agent.py
│   ├── action_planning_agent.py
│   ├── .env.example                 # Copy to .env and fill in keys
│   └── README.md
├── phase2/                          # Agentic workflow built on the Phase 1 library
│   ├── workflow_agents/
│   │   ├── __init__.py
│   │   └── base_agents.py           # Synced copy of phase1/.../base_agents.py
│   ├── agentic_workflow.py          # The end-to-end workflow
│   ├── Product-Spec-Email-Router.txt# Sample product spec processed by the workflow
│   └── README.md
├── .gitignore
└── README.md                        # This file
```

> **Note:** `base_agents.py` is intentionally duplicated in both phases so each runs
> standalone. After editing the Phase 1 copy, re-sync it:
> ```bash
> cp phase1/workflow_agents/base_agents.py phase2/workflow_agents/base_agents.py
> ```

---

## The agent library (Phase 1)

`workflow_agents/base_agents.py` provides seven reusable agent classes:

| Agent | Purpose |
|-------|---------|
| `DirectPromptAgent` | Sends a prompt straight to Claude with no system prompt. |
| `AugmentedPromptAgent` | Adds a persona system prompt to shape the response. |
| `KnowledgeAugmentedPromptAgent` | Answers strictly from supplied knowledge, not the model's own. |
| `RAGKnowledgePromptAgent` | Chunks + embeds a document (OpenAI embeddings), retrieves relevant chunks, answers with Claude. |
| `EvaluationAgent` | Runs a worker agent, judges its output against criteria, and loops with correction feedback until it passes. |
| `RoutingAgent` | Embeds the input and each route's description, routes to the highest cosine-similarity agent. |
| `ActionPlanningAgent` | Extracts an ordered list of steps from a goal prompt. |

---

## The workflow (Phase 2)

`agentic_workflow.py` wires the agents into a project-management pipeline:

1. **Action planning** — `ActionPlanningAgent` breaks the workflow prompt into ordered steps.
2. **Routing** — each step is routed (by embedding similarity) to the right specialist:
   - **Product Manager** → defines user stories from the spec.
   - **Program Manager** → groups stories into features.
   - **Development Engineer** → defines engineering tasks.
   Each specialist is a `KnowledgeAugmentedPromptAgent` paired with an `EvaluationAgent`
   that validates its output against a required structure before accepting it.
3. **Assembly** — the final step compiles all validated artifacts into one development plan.

### Design decisions worth knowing

These address real failure modes found while building the workflow:

- **Context chaining.** Each step's validated output is fed forward as context to later
  steps, so user stories inform features and features inform tasks — keeping the whole plan
  grounded in the actual product spec rather than generic boilerplate.
- **Deterministic final compile.** The "compile everything into a plan" step assembles the
  already-validated artifacts in code instead of asking an agent to regenerate them. This
  costs no extra tokens and cannot truncate.
- **Per-story task generation.** The Development Engineer generates tasks one user story at
  a time (each a bounded evaluation loop), then assembles them. No single model call is ever
  asked to emit tasks for the whole product at once, which would exceed the output limit.

---

## Setup

Requires Python 3.10+.

```bash
# from the repo root
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install anthropic openai numpy pandas python-dotenv
```

### Environment variables

Copy the example file and fill in your keys (the `.env` is gitignored — never commit it):

```bash
cp phase1/.env.example phase1/.env
cp phase1/.env.example phase2/.env   # phase2 needs its own copy
```

| Variable | Required by | Notes |
|----------|-------------|-------|
| `ANTHROPIC_API_KEY` | All chat-based agents | Claude. |
| `ANTHROPIC_MODEL` | All chat-based agents | Defaults to `claude-sonnet-4-6`. |
| `OPENAI_API_KEY` | `RoutingAgent`, `RAGKnowledgePromptAgent` | Embeddings only. |

---

## Running

### Phase 1 — test the agents individually

From `phase1/` (with the venv active):

```bash
cd phase1
python direct_prompt_agent.py
python augmented_prompt_agent.py
python knowledge_augmented_prompt_agent.py
python rag_knowledge_prompt_agent.py
python evaluation_agent.py
python routing_agent.py
python action_planning_agent.py
```

### Phase 2 — run the full workflow

```bash
cd phase2
python agentic_workflow.py
```

The workflow prints each step as it's planned, routed, and evaluated, then prints the final
consolidated development plan. The per-story task generation makes many bounded calls, so a
full run takes a few minutes.

---

## Notes

- Anthropic has no embeddings API, so OpenAI is used wherever embeddings are needed. The
  embeddings provider can be swapped (e.g. for Voyage) without changing the chat agents.
- `RAGKnowledgePromptAgent` writes scratch CSVs (`chunks-*.csv`, `embeddings-*.csv`) in the
  working directory; these are gitignored.
