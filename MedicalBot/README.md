# MedDietQA — Food & Drug Interaction Assistant

`medDietQA.py` is a Streamlit application that helps patients and clinicians explore diet considerations for prescription medications. It combines authoritative medical sources, retrieval-augmented generation (RAG), and OpenAI models to surface concise food and drink guidance.

## Key Features

- Detects the drug mentioned in a free-form question (handles common brand aliases)
- Resolves RxNorm identifiers via the NIH RxNav API
- Collects consumer-friendly medication guidance from MedlinePlus, with NHS UK fallback content
- Extracts food/drink interaction text with focused HTML parsing
- Builds or reuses a Chroma vector store for each drug to power retrieval
- Generates conversational answers using LangChain's RAG pipeline and `gpt-4o`

---

## Architecture

### Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUESTION                            │
│           "Can I drink alcohol on metformin?"                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                ┌────────────────────────┐
                │  1. Drug Name          │
                │     Extraction         │
                │  (heuristic + aliases) │
                └───────────┬────────────┘
                            │  "metformin"
                            ▼
                ┌────────────────────────┐
                │  2. RxNav Lookup       │
                │  (exact → approximate) │
                └───────────┬────────────┘
                            │  RxCUI: "6809"
                            ▼
          ┌─────────────────────────────────────┐
          │  3. Medical Source Resolution        │
          │                                     │
          │  MedlinePlus Connect (JSON → HTML)  │
          │         │                           │
          │    [< 10 words?]                    │
          │         ▼                           │
          │  NHS Medicines (guess → search)     │
          └─────────────────┬───────────────────┘
                            │  Consumer page URL
                            ▼
                ┌────────────────────────┐
                │  4. Content Scraping   │
                │  (3-pass extraction)   │
                └───────────┬────────────┘
                            │  Diet/food text
                            ▼
                ┌────────────────────────┐
                │  5. RAG Pipeline       │
                │  Chunk → Embed →       │
                │  ChromaDB → Retrieve → │
                │  GPT-4o Answer         │
                └───────────┬────────────┘
                            │
                            ▼
                ┌────────────────────────┐
                │  6. Response + Source   │
                │     + Disclaimer       │
                └────────────────────────┘
```

### Component Breakdown

#### 1. Drug Name Extraction (`clean_drug_candidate`)

The simplest layer—intentionally heuristic-based to keep latency low:

| Step | Action | Example |
|------|--------|---------|
| Tokenize | Split query into words | `["Can", "I", "drink", "alcohol", "on", "metformin?"]` |
| Extract | Take the **last token** | `"metformin?"` |
| Normalize | Strip non-alphanumeric chars, lowercase | `"metformin"` |
| Alias Map | Check `NAME_ALIASES` dictionary | `"tylenol"` → `"acetaminophen"` |

Currently supported aliases: `paracetamol`, `tylenol`, `panadol`, `glucophage`, `lipitor`.

#### 2. Drug Identification — RxNav (`rxcui_for_name`)

Uses the **NIH RxNav REST API** to resolve the drug name to an RxCUI (RxNorm Concept Unique Identifier), which is the standard identifier used across US medical systems.

```
Exact Match ──► /REST/rxcui.json?name=metformin
     │
  [no match]
     │
     ▼
Approximate ──► /REST/approximateTerm.json?term=metformin&maxEntries=3
                (handles misspellings, brand names, international names)
                Prefers tty="IN" (ingredient) when available
```

#### 3. Medical Source Resolution (Multi-Source, Multi-Fallback)

The system resolves a consumer-facing drug information page through a **cascading fallback chain**:

```
                    ┌───────────────────────────────┐
                    │  fetch_medlineplus_consumer_   │
                    │  info(drug_name)               │
                    └───────────┬───────────────────┘
                                │
               ┌────────────────┼────────────────────┐
               ▼                ▼                    ▼
        RxCUI found?     Preferred Name         Direct Search
              │            Search                    │
              ▼                                      │
     MedlinePlus Connect ◄──────────────────────────┘
      (JSON → HTML fallback)
              │
         [thin/empty?]
              │
              ▼
     NHS Medicines (UK)
      (URL guess → site search)
```

**MedlinePlus Connect** is queried in two formats:
- **JSON** (`knowledgeResponseType=application/json`): URLs are extracted via regex — avoids brittle HTML parsing.
- **HTML** (`knowledgeResponseType=text/html`): BeautifulSoup scrapes `<a>` tags as a fallback.

**URL selection** prioritizes specificity: `/druginfo/meds/` → `/druginfo/` → first available link.

#### 4. Content Scraping Strategy

Both MedlinePlus and NHS scrapers use a **3-pass extraction** strategy to maximize recall while keeping content focused:

| Pass | Target | Method | When Used |
|------|--------|--------|-----------|
| **Pass 0** | Explicit Q&A headings | Match `<h2>`–`<h4>` against phrases like *"special dietary instructions"*, *"food"*, *"alcohol"* | Always tried first |
| **Pass 1** | Generic keyword headings | Match headings against broader keywords (`avoid`, `food`, `diet`, `drink`, `grapefruit`, `meal`) | If Pass 0 yields nothing |
| **Pass 2** | Keyword paragraphs | Scan all `<p>` tags for sentences containing interaction keywords | Last resort fallback |

Each pass collects text from `<p>` and `<li>` elements between the matched heading and the next heading of equal or higher level. Results are **deduplicated** while preserving order.

#### 5. RAG Pipeline

Once authoritative text is collected, it flows through a standard **Retrieval-Augmented Generation** pipeline:

```
Scraped Text
     │
     ▼
CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
     │
     ▼
OpenAI Embeddings (text-embedding-3-small)
     │
     ▼
ChromaDB (persist per drug in chroma_dbs/<drug_name>/)
     │
     ▼
Retriever (top-k=6 chunks)
     │
     ▼
RAG Prompt (LangChain Hub "rlm/rag-prompt" or hardcoded fallback)
     │
     ▼
GPT-4o (ChatOpenAI)
     │
     ▼
Answer + Source Citation + Medical Disclaimer
```

**Key design choices:**
- **Chunk size of 500** with 50-char overlap balances granularity with context preservation.
- **k=6 retrieval** ensures comprehensive coverage for drugs with multiple dietary considerations.
- **Per-drug vector store persistence** means the second question about the same drug skips all embedding work.
- **Prompt fallback**: If LangChain Hub is unreachable, a hardcoded prompt template ensures the app never fails silently.

### Caching Layer

All network-bound functions are decorated with `@st.cache_data(ttl=86400)` (24-hour TTL):

| Function | What's Cached |
|----------|---------------|
| `rxcui_for_name()` | Drug name → RxCUI mapping |
| `rxnorm_preferred_name()` | RxCUI → preferred name |
| `connect_json_urls_for_rxcui()` | MedlinePlus Connect JSON response URLs |
| `fetch_medlineplus_consumer_link_by_rxcui()` | Final consumer page URL |
| `medlineplus_search_for_druginfo()` | MedlinePlus search results |
| `fetch_medlineplus_consumer_info()` | Top-level resolution result |
| `extract_food_advice_from_medlineplus()` | Scraped dietary text |
| `nhs_guess_medicine_url()` | NHS URL guess result |
| `nhs_search_medicine_url()` | NHS search result |
| `extract_food_advice_from_nhs()` | Scraped NHS dietary text |

This means a **first query** for a new drug triggers 3–6 API calls and 1–2 page scrapes, while **subsequent queries** for the same drug within 24 hours hit only the vector store.

### Error Handling Philosophy

The app follows a **"never crash, degrade gracefully"** approach:

- Every external API call is wrapped in `try/except` — failures silently fall through to the next source.
- If no dietary text can be extracted from any source, the user receives a clear error message with the URL that was found (if any), so they can investigate manually.
- The LangChain Hub prompt pull has a hardcoded fallback template.
- If the drug name can't be detected, the user gets an example of a well-formed query.

---

## Project Structure

```
MedicalBot/
├── medDietQA.py        # Main application (all logic in a single file)
├── chroma_dbs/         # Auto-created: persisted ChromaDB stores per drug
│   ├── metformin/
│   ├── atorvastatin/
│   └── ...
└── README.md           # This file
```

---

## Requirements

- Python 3.9+
- Streamlit
- `requests`, `beautifulsoup4`
- `langchain`, `langchain-community`, `langchain-openai`
- `chromadb`
- Valid OpenAI API key with access to `gpt-4o` and `text-embedding-3-small`

Install dependencies via `pip`:

```bash
pip install streamlit requests beautifulsoup4 langchain langchain-community langchain-openai chromadb
```

## Running the App

1. Export your OpenAI key (recommended) or have it ready to paste in the UI:

   ```bash
   export OPENAI_API_KEY=sk-...
   ```

2. Launch Streamlit from the repository root:

   ```bash
   streamlit run MedicalBot/medDietQA.py
   ```

3. In the browser UI:
   - Enter a medication + diet question (e.g., `Can I drink coffee when taking atorvastatin?`).
   - Paste your OpenAI API key when prompted.
   - Submit to receive contextual guidance with cited sources and a disclaimer.

## Notes

- Medical content is sourced live from MedlinePlus and the NHS; network access is required.
- Retrieved pages are cached and vectorized per drug inside the `chroma_dbs/` directory for faster follow-up questions.
- The assistant provides educational information only. Always recommend users confirm advice with a licensed medical professional.

