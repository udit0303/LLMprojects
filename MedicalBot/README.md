# MedDietQA — Food & Drug Interaction Assistant

`medDietQA.py` is a Streamlit application that helps patients and clinicians explore diet considerations for prescription medications. It combines authoritative medical sources, retrieval-augmented generation (RAG), and OpenAI models to surface concise food and drink guidance.

## Key Features

- Detects the drug mentioned in a free-form question (handles common brand aliases)
- Resolves RxNorm identifiers via the NIH RxNav API
- Collects consumer-friendly medication guidance from MedlinePlus, with NHS UK fallback content
- Extracts food/drink interaction text with focused HTML parsing
- Builds or reuses a Chroma vector store for each drug to power retrieval
- Generates conversational answers using LangChain's RAG pipeline and `gpt-4o`

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

