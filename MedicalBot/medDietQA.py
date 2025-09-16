import os
import re
from urllib.parse import urljoin

import streamlit as st
import requests
from bs4 import BeautifulSoup

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain import hub

# =============================
# Constants & Setup
# =============================
CHROMA_DIR = "chroma_dbs"
os.makedirs(CHROMA_DIR, exist_ok=True)

DEFAULT_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; MedDietQA/1.2)"}
REQUEST_TIMEOUT = 20

# Common brand/international aliases → US generic names
NAME_ALIASES = {
    "paracetamol": "acetaminophen",
    "tylenol": "acetaminophen",
    "panadol": "acetaminophen",
    "glucophage": "metformin",
    "lipitor": "atorvastatin",
}

# =============================
# Helpers
# =============================

def clean_drug_candidate(text: str) -> str:
    """Extract a plausible drug name from the user's query.
    Heuristic: take the last word and strip punctuation; apply name aliases.
    """
    parts = text.strip().split()
    token = parts[-1] if parts else ""
    token = re.sub(r"[^A-Za-z0-9\-]", "", token)
    token = token.lower()
    return NAME_ALIASES.get(token, token)


def _pick_best_medlineplus_url(urls: list[str]) -> str | None:
    """Choose the most specific MedlinePlus consumer drug info URL."""
    if not urls:
        return None
    # Prefer /druginfo/meds/ pages
    for u in urls:
        if "/druginfo/meds/" in u:
            return u
    # Next, any /druginfo page
    for u in urls:
        if "/druginfo" in u:
            return u
    return urls[0]

# =============================
# RxNav + MedlinePlus Connect
# =============================

@st.cache_data(ttl=86400)
def rxcui_for_name(drug_name: str) -> str | None:
    """Look up RxCUI (RxNorm concept ID) for a given drug name using RxNav.
    Tries exact match first, then approximateTerm for misspellings/intl names.
    Returns the first RxCUI string or None if not found.
    """
    # Exact match
    try:
        url = "https://rxnav.nlm.nih.gov/REST/rxcui.json"
        resp = requests.get(url, params={"name": drug_name}, headers=DEFAULT_HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        ids = data.get("idGroup", {}).get("rxnormId") or []
        if ids:
            return ids[0]
    except Exception:
        pass

    # Fallback: approximateTerm (helps with brand names, misspellings, international names)
    try:
        approx_url = "https://rxnav.nlm.nih.gov/REST/approximateTerm.json"
        resp = requests.get(approx_url, params={"term": drug_name, "maxEntries": 3}, headers=DEFAULT_HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        candidates = (data.get("approximateGroup", {}) or {}).get("candidate", [])
        # Prefer ingredients (IN) when present
        for c in candidates:
            if c.get("tty") == "IN" and c.get("rxcui"):
                return c.get("rxcui")
        if candidates:
            return candidates[0].get("rxcui")
    except Exception:
        pass

    return None

@st.cache_data(ttl=86400)
def rxnorm_preferred_name(rxcui: str) -> str | None:
    """Return RxNorm preferred name for a given RxCUI (e.g., returns 'Acetaminophen')."""
    try:
        url = f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/properties.json"
        resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json() or {}
        return ((data.get("properties") or {}).get("name"))
    except Exception:
        return None

@st.cache_data(ttl=86400)
def connect_json_urls_for_rxcui(rxcui: str) -> list[str]:
    """Call MedlinePlus Connect asking for JSON, then regex out medlineplus.gov URLs.
    This avoids brittle HTML parsing.
    """
    urls: list[str] = []
    try:
        connect_url = "https://connect.medlineplus.gov/service"
        params = {
            "mainSearchCriteria.v.c": rxcui,
            "mainSearchCriteria.v.cs": "2.16.840.1.113883.6.88",  # RxNorm OID
            "informationRecipient.languageCode.c": "en",
            "knowledgeResponseType": "application/json",
        }
        resp = requests.get(connect_url, params=params, headers=DEFAULT_HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        text = resp.text or ""
        urls = re.findall(r"https://medlineplus\.gov/[A-Za-z0-9_\-/\.]+", text)
        urls = list(dict.fromkeys(urls))  # dedupe
    except Exception:
        pass
    return urls

@st.cache_data(ttl=86400)
def fetch_medlineplus_consumer_link_by_rxcui(rxcui: str) -> str | None:
    # Try JSON first
    urls = connect_json_urls_for_rxcui(rxcui)
    pick = _pick_best_medlineplus_url(urls)
    if pick:
        return pick

    # Fallback: HTML scrape of Connect service
    try:
        connect_url = "https://connect.medlineplus.gov/service"
        params = {
            "mainSearchCriteria.v.c": rxcui,
            "mainSearchCriteria.v.cs": "2.16.840.1.113883.6.88",
            "informationRecipient.languageCode.c": "en",
            "knowledgeResponseType": "text/html",
        }
        resp = requests.get(connect_url, params=params, headers=DEFAULT_HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        html_urls = [urljoin(resp.url, a["href"]) for a in soup.find_all("a", href=True)]
        html_urls = [u for u in html_urls if u.startswith("https://medlineplus.gov/")]
        return _pick_best_medlineplus_url(html_urls)
    except Exception:
        return None

@st.cache_data(ttl=86400)
def medlineplus_search_for_druginfo(name: str) -> str | None:
    """Search MedlinePlus (web service) and return first consumer med page.
    """
    try:
        ws_url = "https://wsearch.nlm.nih.gov/ws/query"
        params = {"db": "healthTopics", "term": name}
        resp = requests.get(ws_url, params=params, headers=DEFAULT_HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "xml")
        urls = [u.get_text(strip=True) for u in soup.find_all("url")]
        urls = [u for u in urls if u.startswith("https://medlineplus.gov/")]
        return _pick_best_medlineplus_url(urls)
    except Exception:
        return None

@st.cache_data(ttl=86400)
def fetch_medlineplus_consumer_info(drug_name: str) -> str | None:
    """High-level resolver: drug name -> RxCUI -> MedlinePlus consumer page link.
    Falls back to site search if Connect yields a generic page or no direct hit.
    Returns a URL or None.
    """
    rxcui = rxcui_for_name(drug_name)
    if not rxcui:
        # No RxCUI found; try searching directly
        return medlineplus_search_for_druginfo(drug_name)

    link = fetch_medlineplus_consumer_link_by_rxcui(rxcui)
    if link and "/druginfo/meds/" in link:
        return link

    # Try searching by the RxNorm preferred name
    pref = rxnorm_preferred_name(rxcui)
    if pref:
        url = medlineplus_search_for_druginfo(pref)
        if url:
            return url

    # Last resort: search by original name
    return medlineplus_search_for_druginfo(drug_name)

# =============================
# Scraping consumer page for diet/food interactions
# =============================

@st.cache_data(ttl=86400)
def extract_food_advice_from_medlineplus(drug_info_url: str) -> str:
    """Download the MedlinePlus page and extract sections that mention diet/food/drink.
    Strategy:
      1) Scan headings (h2-h4) with keywords and collect text until next heading.
      2) If nothing found, scan all paragraphs for sentences with keywords as a fallback.
      3) Extra targeting for common FAQ-style headings like 'What special dietary instructions should I follow?'.
    """
    resp = requests.get(drug_info_url, headers=DEFAULT_HEADERS, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    section_keywords = {"avoid", "food", "diet", "dietary", "eat", "drink", "grapefruit", "alcohol", "with food", "without food", "meal"}
    relevant_text: list[str] = []

    def collect_until_next_heading(start_tag):
        for tag in start_tag.find_all_next():
            if getattr(tag, "name", None) in ["h2", "h3", "h4"]:
                break
            if getattr(tag, "name", None) in ["p", "li"]:
                text = tag.get_text(" ", strip=True)
                if text:
                    relevant_text.append(text)

    # Pass 0: explicitly look for common Q&A headings
    explicit_phrases = [
        "special dietary instructions",
        "food",
        "drink",
        "alcohol",
        "with food",
        "without food",
        "grapefruit",
    ]
    for h in soup.find_all(["h2", "h3", "h4"]):
        title = h.get_text(" ", strip=True).lower()
        if any(p in title for p in explicit_phrases):
            collect_until_next_heading(h)
            if relevant_text:
                break

    # Pass 1: generic keyword headings
    if not relevant_text:
        for header in soup.find_all(["h2", "h3", "h4"]):
            title = header.get_text(strip=True).lower()
            if any(k in title for k in section_keywords):
                collect_until_next_heading(header)
                if relevant_text:
                    break

    # Pass 2: keyword paragraphs
    if not relevant_text:
        for p in soup.find_all("p"):
            text = p.get_text(" ", strip=True)
            lt = text.lower()
            if any(k in lt for k in ["alcohol", "grapefruit", "food", "diet", "take with", "take without", "with meals", "after meals"]):
                relevant_text.append(text)

    # Deduplicate while preserving order
    deduped = list(dict.fromkeys(relevant_text))
    return "\n\n".join(deduped).strip()

# =============================
# NHS fallback (UK Medicines)
# =============================

@st.cache_data(ttl=86400)
def nhs_guess_medicine_url(name: str) -> str | None:
    """Guess the NHS medicine URL based on canonical pattern."""
    slug = re.sub(r"[^a-z0-9\-]", "-", name.lower())
    slug = re.sub(r"-+", "-", slug).strip("-")
    url = f"https://www.nhs.uk/medicines/{slug}/"
    try:
        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        if r.status_code == 200 and "/medicines/" in r.url:
            return r.url
    except Exception:
        pass
    return None

@st.cache_data(ttl=86400)
def nhs_search_medicine_url(name: str) -> str | None:
    """Search NHS site and return first /medicines/ URL."""
    try:
        search_url = "https://www.nhs.uk/search/"
        r = requests.get(search_url, params={"q": name}, headers=DEFAULT_HEADERS, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            href_abs = href if href.startswith("http") else urljoin(r.url, href)
            if "/medicines/" in href_abs:
                return href_abs
    except Exception:
        pass
    return None

@st.cache_data(ttl=86400)
def extract_food_advice_from_nhs(nhs_url: str) -> str:
    """Extract 'Food and drink' and alcohol-related guidance from an NHS medicine page."""
    r = requests.get(nhs_url, headers=DEFAULT_HEADERS, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    relevant_text: list[str] = []
    keywords = ["food", "drink", "alcohol", "with food", "without food", "meal", "meals"]

    # Headings first
    for h in soup.find_all(["h2", "h3", "h4"]):
        title = h.get_text(" ", strip=True).lower()
        if any(k in title for k in keywords):
            for tag in h.find_all_next():
                if getattr(tag, "name", None) in ["h2", "h3", "h4"]:
                    break
                if getattr(tag, "name", None) in ["p", "li"]:
                    txt = tag.get_text(" ", strip=True)
                    if txt:
                        relevant_text.append(txt)
            if relevant_text:
                break

    # Paragraph fallback
    if not relevant_text:
        for p in soup.find_all("p"):
            txt = p.get_text(" ", strip=True)
            lt = txt.lower()
            if any(k in lt for k in keywords):
                relevant_text.append(txt)

    deduped = list(dict.fromkeys(relevant_text))
    return "\n\n".join(deduped).strip()

# =============================
# RAG Utilities
# =============================

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_or_create_chroma(drug_name: str, document_text: str, embeddings: OpenAIEmbeddings) -> Chroma:
    db_path = os.path.join(CHROMA_DIR, drug_name.lower())
    if os.path.exists(db_path) and os.listdir(db_path):
        return Chroma(embedding_function=embeddings, persist_directory=db_path)
    docs = CharacterTextSplitter(chunk_size=500, chunk_overlap=50).create_documents([document_text])
    vs = Chroma.from_documents(docs, embeddings, persist_directory=db_path)
    vs.persist()
    return vs

# =============================
# Core RAG Flow
# =============================

def generate_response(query_text: str, openai_api_key: str) -> str:
    possible_drug = clean_drug_candidate(query_text)
    if not possible_drug:
        return "❌ I couldn't detect a medicine name in your question. Try: 'Can I drink alcohol on metformin?'"

    # 1) Try MedlinePlus first
    medlineplus_url = fetch_medlineplus_consumer_info(possible_drug)
    source_label = None
    drug_text = ""

    if medlineplus_url:
        try:
            drug_text = extract_food_advice_from_medlineplus(medlineplus_url)
            source_label = f"MedlinePlus — {medlineplus_url}"
        except Exception:
            drug_text = ""

    # 2) Fallback to NHS if MLP text is thin
    if not drug_text or len(drug_text.split()) < 10:
        nhs_url = nhs_guess_medicine_url(possible_drug) or nhs_search_medicine_url(possible_drug)
        if nhs_url:
            try:
                nhs_text = extract_food_advice_from_nhs(nhs_url)
                if len(nhs_text.split()) >= len(drug_text.split()):
                    drug_text = nhs_text
                    source_label = f"NHS Medicines — {nhs_url}"
            except Exception:
                pass

    if not drug_text:
        msg = "❌ Couldn't extract reliable food/drink guidance. Try another phrasing, or a different drug."
        if medlineplus_url:
            msg += f"\n(We found a page at {medlineplus_url} but it didn't include a clear 'Food/Drink' section.)"
        return msg

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
    vectorstore = get_or_create_chroma(possible_drug, drug_text, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    # Pull a RAG prompt from the hub with a graceful fallback
    try:
        prompt = hub.pull("rlm/rag-prompt")
    except Exception:
        from langchain.prompts import PromptTemplate
        prompt = PromptTemplate.from_template(
            """
You are a careful assistant. Use the context to answer the user's question about food/drug interactions.
If you don't know, say so. Be concise, actionable, and avoid speculation.

Context:
{context}

Question: {question}
Answer:
"""
        )

    llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke(query_text)

    disclaimer = (
        "\n\n—\n**Source:** " + (source_label or "Unknown") + ". This is not medical advice; consult your clinician or pharmacist for personal guidance."
    )
    return f"{answer}{disclaimer}"

# =============================
# Streamlit UI
# =============================

st.set_page_config(page_title="💊 Food & Drug Q/A Bot", layout="centered")
st.title("💊 What Can I Eat With My Medicine?")
st.markdown(
    "Ask about food or drink interactions with a medication. Example:\n\n"
    "`Can I drink alcohol on metformin?`"
)

query_text = st.text_input("Your question:")

result = None
with st.form("qa_form", clear_on_submit=False):
    openai_api_key = st.text_input("🔐 OpenAI API Key", type="password", disabled=not query_text)
    submitted = st.form_submit_button("Submit", disabled=not query_text)
    if submitted:
        with st.spinner("Fetching medical guidance..."):
            result = generate_response(query_text, openai_api_key)

if result:
    st.success(result)
