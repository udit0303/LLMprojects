# Streamlit Q&A Bot

This project contains a simple Streamlit web app (`StreamListExample1/StreamlitExample.py`) that lets you upload a text file and ask questions about its contents. The app uses LangChain, OpenAI embeddings, and a retriever-augmented generation chain to produce answers.

## Features

- Upload `.txt` documents through the Streamlit UI
- Split the document into overlapping chunks for retrieval
- Embed the chunks with the `text-embedding-3-small` OpenAI model
- Retrieve relevant context using a Chroma vector store
- Generate answers with the `gpt-4o` ChatOpenAI model and LangChain's RAG prompt template

## Requirements

- Python 3.9+
- Streamlit
- langchain, langchain-community, langchain-openai
- Chroma vector store dependencies
- Access to the OpenAI API (`sk-` prefixed key)

You can install the dependencies with:

```bash
pip install -r requirements.txt
```

If you do not have a `requirements.txt` yet, install the packages manually:

```bash
pip install streamlit langchain langchain-community langchain-openai chromadb
```

## Usage

1. Set the `OPENAI_API_KEY` environment variable or have your key ready.
2. Start the Streamlit app:

   ```bash
   streamlit run StreamListExample1/StreamlitExample.py
   ```

3. Upload a `.txt` file, enter a question, provide your OpenAI API key in the form, and submit.
4. The app will display the generated answer in the info box.

## Notes

- The form disables input fields until you upload a file and enter a question, reducing accidental submissions.
- The OpenAI API key must start with `sk-`; otherwise the app will not trigger a request.
- Uploaded files are processed in memory only during the active session.

