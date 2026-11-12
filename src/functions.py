"""Your reusable functions live here.

The rule from the brief: Python logic goes in `.py` files, and the notebooks
hold the narrative. The app imports the same functions, so the text it receives
goes through exactly the same cleaning as your training data.

To use this module from a notebook in `notebooks/`:

    import sys
    sys.path.append("..")
    from src.functions import *

and from the app in `app/`, the same two lines with `sys.path.append(".")` when
you run `streamlit run app/streamlit_app.py` from the repo root.

The stubs below only show the shape. Rename them, change the arguments, delete
the ones for the other track, write your own.
"""

"""Reusable logic for Trail Scout, imported by the notebooks and the Streamlit app.
 
The pipeline from raw Wikivoyage wikitext to searchable chunks:
 
    load_articles()          raw JSONL  -> one row per article
    split_sections()         wikitext   -> [(section name, section wikitext), ...]
    clean_wikitext()         wikitext   -> plain text (listings kept as readable lines)
    build_chunks()           articles   -> one row per chunk, with the metadata answers cite
"""

import re
import json
import html
import time
import pandas as pd
import numpy as np

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed" 
MODELS = ROOT / "models"

ARTICLES_FILE = RAW / "wikivoyage_balkans.jsonl"
MANIFEST_FILE = RAW / "wikivoyage_balkans_manifest.json"
CHUNKS_FILE = PROCESSED / "chunks.jsonl"
EVAL_FILE = ROOT / "data" / "eval_questions.csv"
BASELINE_ANSWERS_FILE = ROOT / "data" / "baseline_llm_answers.csv" 

RANDOM_STATE = 42
LLM_MODEL = "gpt-5.4-nano"                                    


PRICES = {"gpt-5.4-nano": (0.20, 1.25)}      # US dollars per million tokens (input, output), October 2026
usage_log: list[dict] = []

def cost(input_tokens, output_tokens):
    price_in, price_out = PRICES[LLM_MODEL]
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000

def track(response, step):
    """Record the tokens of a response under the name of a step, and return the response."""
    usage_log.append({"step": step, "input": response.usage.prompt_tokens, "output": response.usage.completion_tokens})
    return response
 

# Article types that are about nature by definition. Other types, including
# "extraregion" (Wikivoyage's areas that cross region boundaries: the Prokletije
# mountains, but also the island of Pag), count as nature only through their
# share of nature words.
NATURE_TYPES = {"park", "ruralarea"}

# Words that signal outdoor and mountain content. Counted per 1,000 words to
# score how much an article (or a chunk) is about nature.
NATURE_TERMS = re.compile(
    r"\b(hik(?:e|es|ed|ing|er|ers)|trek\w*|trails?|mountains?|mountaineer\w*|peaks?|summits?|"
    r"ridges?|national parks?|nature parks?|canyons?|gorges?|lakes?|waterfalls?|caves?|"
    r"climb\w*|ski(?:ing|s)?|forests?|glacial|valleys?|camp(?:ing|site|sites)?|huts?|"
    r"via ferrata|rafting|meadows?|wildlife|bears?)\b",
    re.IGNORECASE,
)



# Section names of a Wikivoyage article that hold no travel content of their own.
SKIPPED_SECTIONS = {"go next", "connect"}



# Listing templates ({{see|name=...|content=...}}) and the fields kept from them.
# Phone, email, url and coordinates are dropped: they are not needed to answer
# a traveller's question, and they keep business contact details out of the prompts.
LISTING_TYPES = {"see", "do", "buy", "eat", "drink", "sleep", "go", "listing"}
LISTING_FIELDS = [("price", "Price"), ("hours", "Hours"), ("directions", "Directions"),
                  ("address", "Address")]



# Small templates that wrap a value with a unit or a currency: {{km|5}} -> "5 km".
UNIT_TEMPLATES = {
    "eur": "€{}", "all": "{} lek", "km": "{} km", "m": "{} m", "ft": "{} ft",
    "kmh": "{} km/h", "ha": "{} ha", "km2": "{} km²", "celsius": "{}°C",
}



# --- Loading --------------------------------------------------------------------
 
def load_articles(path: Path = ARTICLES_FILE) -> pd.DataFrame:
    """Load the downloaded Wikivoyage articles into a DataFrame, one row per article.
 
    The downloader reads the article type from status templates such as
    {{usablepark}}; extra-hierarchical regions have no such template, so they
    are recognised here from {{extraregion}}. Articles with neither get "unknown".
    """
    df = pd.read_json(path, lines=True)
    is_extraregion = df["wikitext"].str.contains(r"\{\{\s*extraregion", case=False, regex=True)
    df.loc[is_extraregion, "article_type"] = "extraregion"
    df["article_type"] = df["article_type"].fillna("unknown")
    df["status"] = df["status"].fillna("unknown")
    df["last_edited"] = pd.to_datetime(df["last_edited"])
    return df



# --- Splitting into sections ----------------------------------------------------
 
HEADING_2 = re.compile(r"^==\s*([^=].*?)\s*==\s*$", re.MULTILINE)
 
 
def split_sections(wikitext: str) -> list[tuple[str, str]]:
    """Split an article at its level-2 headings.
 
    Returns (section name, raw wikitext) pairs. The text before the first
    heading is the article's introduction and is named "Introduction".
    Level-3 headings (=== By car ===) stay inside their section.
    """
    parts = HEADING_2.split(wikitext or "")
    sections = [("Introduction", parts[0])]
    sections += [(name.strip(), body) for name, body in zip(parts[1::2], parts[2::2])]
    return sections



# --- Cleaning wikitext ----------------------------------------------------------
 
INNERMOST_TEMPLATE = re.compile(r"\{\{([^{}]*)\}\}")
INTERNAL_LINK = re.compile(r"\[\[([^\[\]|]+)(?:\|([^\[\]]*))?\]\]")
EXTERNAL_LINK = re.compile(r"\[(?:https?:)?//[^\s\]]+(?:\s+([^\]]*))?\]")
TABLE = re.compile(r"\{\|.*?\|\}", re.DOTALL)
REF = re.compile(r"<ref[^>/]*/>|<ref[^>]*>.*?</ref>", re.DOTALL | re.IGNORECASE)
COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
HEADING_3 = re.compile(r"^={3,}\s*(.*?)\s*={3,}\s*$", re.MULTILINE)
# Phone numbers: international ones (+40 268 ...), or any number after "tel", "phone", ...
PHONE_INTERNATIONAL = re.compile(r"\+\d[\d ()./\-]{6,}\d")
PHONE_AFTER_WORD = re.compile(r"(\b(?:tel|phone|mobile|gsm|fax)\b\.?:?\s*)\(?\d[\d ()./\-]{6,}\d", re.IGNORECASE)
EMAIL = re.compile(r"[\w.+\-]+@[\w\-]+\.[\w.\-]+")
 
 
def _remove_file_links(text: str) -> str:
    """Remove [[File:...]] and [[Image:...]] links, whose captions can contain nested links."""
    out, i = [], 0
    pattern = re.compile(r"\[\[\s*(?:File|Image):", re.IGNORECASE)
    while (match := pattern.search(text, i)):
        out.append(text[i:match.start()])
        depth, j = 0, match.start()
        while j < len(text):                 # walk to the matching closing ]]
            if text.startswith("[[", j):
                depth, j = depth + 1, j + 2
            elif text.startswith("]]", j):
                depth, j = depth - 1, j + 2
                if depth == 0:
                    break
            else:
                j += 1
        i = j
    out.append(text[i:])
    return "".join(out)
 
 
def _template_params(body: str) -> tuple[str, list[str], dict[str, str]]:
    """Split a template body into its name, positional values and named values."""
    name, *raw_params = body.split("|")
    positional, named = [], {}
    for param in raw_params:
        key, sep, value = param.partition("=")
        if sep and re.fullmatch(r"\s*[\w ]+\s*", key):
            named[key.strip().lower()] = value.strip()
        else:
            positional.append(param.strip())
    return name.strip().lower(), positional, named
 
 
def _listing_to_text(params: dict[str, str]) -> str:
    """Turn a listing's fields into one readable line, keeping its last-checked date."""
    name = params.get("name", "")
    if params.get("alt"):
        name += f" ({params['alt']})"
    parts = [f"{name}:" if name else "", params.get("content", "")]
    parts += [f"{label}: {params[key]}." for key, label in LISTING_FIELDS if params.get(key)]
    if params.get("lastedit"):
        parts.append(f"(Last checked {params['lastedit']}.)")
    return " ".join(p for p in parts if p).strip()
 
 
def _render_template(match: re.Match) -> str:
    """Replace one template with the text it stands for, or with nothing."""
    name, positional, named = _template_params(match.group(1))
    if name in LISTING_TYPES:
        return "\n" + _listing_to_text(named) + "\n"
    if name == "marker":
        return named.get("name", "")
    if name in UNIT_TEMPLATES and positional:
        return UNIT_TEMPLATES[name].format(positional[0])
    if name == "convert" and len(positional) >= 2:
        return f"{positional[0]} {positional[1]}"
    return ""          # layout and maintenance templates: banners, maps, status, dead link, ...
 
 
def clean_wikitext(wikitext: str) -> str:
    """Convert Wikivoyage wikitext into plain text for search and for the LLM.
 
    Links keep their visible text, listings become one line each with their
    price, hours and last-checked date, unit templates keep their value, and
    everything else (images, maps, banners, tables, references, HTML) is removed.
    Phone numbers and email addresses written into the text are masked, so no
    contact details are sent to the LLM.
    Level-3 headings become a line ending with a colon, so "By car:" stays
    readable; headings with nothing under them are dropped.
    """
    text = COMMENT.sub("", wikitext or "")
    text = REF.sub("", text)
    text = _remove_file_links(text)
    text = re.sub(r"\[\[\s*Category:[^\]]*\]\]", "", text, flags=re.IGNORECASE)
    # Links before templates: a link's "|" would otherwise split a template's fields.
    text = INTERNAL_LINK.sub(lambda m: m.group(2) or m.group(1), text)
    text = EXTERNAL_LINK.sub(lambda m: m.group(1) or "", text)
    while INNERMOST_TEMPLATE.search(text):              # innermost first: {{see|...{{eur|5}}...}}
        text = INNERMOST_TEMPLATE.sub(_render_template, text)
    text = TABLE.sub("", text)
    text = re.sub(r"\{\{[^|}\n]*\|?|\}\}|\[\[|\]\]", "", text)   # stray brackets of broken markup
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = HTML_TAG.sub("", text)
    text = html.unescape(text).replace("\xa0", " ")    # &nbsp; &amp; ... -> characters
    text = EMAIL.sub("[email]", text)
    text = PHONE_INTERNATIONAL.sub("[phone]", text)
    text = PHONE_AFTER_WORD.sub(r"\1[phone]", text)
    text = HEADING_3.sub(lambda m: f"\n\x00{m.group(1)}:", text)  # \x00 marks a heading line
    text = re.sub(r"^\x00[^\n]*\n\s*(?=\x00|\Z)", "", text + "\n", flags=re.MULTILINE)  # empty headings
    text = text.replace("\x00", "")
    text = re.sub(r"'{2,}", "", text)                    # bold and italic quotes
    text = re.sub(r"^[*#:;]+\s*", "- ", text, flags=re.MULTILINE)   # list markers
    text = re.sub(r"^-\s*$", "", text, flags=re.MULTILINE)          # list items left empty
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()



 
# --- Scoring and chunking -------------------------------------------------------
 
def count_words(text: str) -> int:
    """Number of whitespace-separated words in a text."""
    return len(text.split())
 
 
def nature_terms_per_1k(text: str) -> float:
    """How many nature and outdoor words a text has per 1000 words (0 for an empty text)."""
    n_words = count_words(text)
    return 1000 * len(NATURE_TERMS.findall(text)) / n_words if n_words else 0.0
 
 
def chunk_text(text: str, max_words: int, overlap: int) -> list[str]:
    """Split a text into windows of at most max_words words, consecutive windows sharing `overlap` words.
 
    A text no longer than max_words comes back as a single chunk.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text] if words else []
    step = max_words - overlap
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words) - overlap, step)]
 

def build_chunks(articles: pd.DataFrame, max_words: int = 105, overlap: int = 21,
                 min_words: int = 15) -> pd.DataFrame:
    """Turn articles into searchable chunks, one row per chunk, with the metadata answers cite.
 
    Each article is split into sections, each section is cleaned, sections in
    SKIPPED_SECTIONS or with fewer than `min_words` words are dropped, and long
    sections are cut into overlapping windows. `embed_text` prefixes the chunk
    with its article and section, so a chunk such as "Entry is €5 per day" still
    says which park it is about when it is embedded or searched.
    """
    rows = []
    for article in articles.itertuples(index=False):
        for section, raw in split_sections(article.wikitext):
            if section.lower() in SKIPPED_SECTIONS:
                continue
            text = clean_wikitext(raw)
            if count_words(text) < min_words:
                continue
            for i, chunk in enumerate(chunk_text(text, max_words, overlap)):
                rows.append({
                    "chunk_id": f"wv-{len(rows)}",
                    "title": article.title,
                    "country": article.country,
                    "article_type": article.article_type,
                    "section": section,
                    "url": article.url,
                    "last_edited": article.last_edited,
                    "text": chunk,
                    "embed_text": f"{article.title} ({article.country}) - {section}\n{chunk}",
                    "n_words": count_words(chunk),
                })
    return pd.DataFrame(rows)
 
 
def save_chunks(chunks: pd.DataFrame, path: Path = CHUNKS_FILE) -> None:
    """Write the chunks to JSON Lines, creating the folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks.to_json(path, orient="records", lines=True, force_ascii=False, date_format="iso")
 
 
def load_manifest(path: Path = MANIFEST_FILE) -> dict:
    """Read the download manifest: date, licence and counts."""
    return json.loads(path.read_text(encoding="utf-8"))




# --- The baseline: the LLM alone ------------------------------------------------

# The baseline's instructions: the same refusal rule the RAG prompt will have,
# so the comparison only changes one thing, the documents.
BASELINE_INSTRUCTIONS = (
    "You are a travel assistant for hikers and nature travellers who want to visit the Balkans. "
    "Answer the traveller's question in at most 3 sentences. "
    "If you do not know the answer, say 'I don't know' instead of guessing."
)


def load_eval_set(path: Path = EVAL_FILE) -> pd.DataFrame:
    """Load the evaluation questions; empty cells (out-of-scope questions) become empty strings."""
    return pd.read_csv(path).fillna("")


def ask_llm_alone(client, question: str, step: str = "baseline") -> dict:
    """Ask the LLM a question with no documents, and return its answer with its tokens and cost."""
    messages = [
        {"role": "system", "content": BASELINE_INSTRUCTIONS},
        {"role": "user", "content": question},
    ]
    response = track(client.chat.completions.create(model=LLM_MODEL, messages=messages), step)
    usage = response.usage
    return {
        "answer": response.choices[0].message.content.strip(),
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "cost_usd": cost(usage.prompt_tokens, usage.completion_tokens),
    }


REFUSAL_PATTERN = re.compile(r"\bi don.?t know\b|\bi do not know\b|\bnot sure\b|\bcan.?t (?:help|book|check)|\bcannot\b|\bunable to\b", re.IGNORECASE)

def run_llm_baseline(client, questions: pd.DataFrame, path: Path = BASELINE_ANSWERS_FILE,
                     rerun: bool = False) -> pd.DataFrame:
    """Run the LLM-alone baseline on every question, or load the saved answers.
 
    The answers are saved to `path` and reused on later runs unless `rerun` is
    True, so restarting the notebook does not pay for the calls again and the
    labels keep matching the answers they were given for. `looks_like_refusal`
    is a first guess to speed up labelling, not a label.
    """
    if path.exists() and not rerun:
        return pd.read_csv(path).fillna("")
    rows = []
    for q in questions.itertuples(index=False):
        rows.append({
            "question_id": q.question_id,
            **ask_llm_alone(client, q.question, step=f"baseline {q.question_id}")})
        print(f"{q.question_id}: {rows[-1]['input_tokens']} input, {rows[-1]['output_tokens']} output tokens")
    answers = pd.DataFrame(rows)
    answers["looks_like_refusal"] = answers["answer"].str.contains(REFUSAL_PATTERN)
    path.parent.mkdir(parents=True, exist_ok=True)
    answers.to_csv(path, index=False)
    return answers



ANSWER_LABELS = {"correct", "partial", "wrong", "refused"}

def labels_summary(results: pd.DataFrame) -> pd.Series:
    """Turn per-question labels into the baseline's scores.
 
    `results` needs the columns `kind`, `label` (one of ANSWER_LABELS), `cites_source`
    (bool), `cost_usd` and `seconds`. On answerable questions: the share correct,
    partly correct, wrong and refused, and grounded accuracy (correct *and*
    citing a source). On out-of-scope questions: the share refused, which should be 1.
    """
    bad = set(results["label"]) - ANSWER_LABELS
    if bad:
        raise ValueError(f"Unknown labels {bad}; use one of {sorted(ANSWER_LABELS)}")
    answerable = results[results["kind"] != "out_of_scope"]
    out_of_scope = results[results["kind"] == "out_of_scope"]
    scores = {f"answerable: {label}": (answerable["label"] == label).mean()
              for label in ["correct", "partial", "wrong", "refused"]}
    scores["answerable: grounded accuracy"] = ((answerable["label"] == "correct") & answerable["cites_source"]).mean()
    scores["out of scope: refused"] = (out_of_scope["label"] == "refused").mean()
    scores["cost per question (USD)"] = results["cost_usd"].mean()
    scores["total cost (USD)"] = results["cost_usd"].sum()
    right = ((results["kind"] != "out_of_scope") & (results["label"] == "correct")) | \
        ((results["kind"] == "out_of_scope") & (results["label"] == "refused"))
    scores["overall accuracy (all questions)"] = right.mean()
    scores["answerable: lenient accuracy (partial = 0.5)"] = (
        (answerable["label"] == "correct").mean() + 0.5 * (answerable["label"] == "partial").mean()
    )

    return pd.Series(scores)



# --- RAG settings: defined once --------------------------------------------------
 
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = MODELS / "chroma"         
COLLECTION_NAME = "wikivoyage"
NATURE_ONLY = {"is_nature": True}

TOP_K = 5
WHERE = None                          
 
REFUSAL = "I don't know based on my guides."
 
SYSTEM = f"""You answer questions from hikers and nature travellers about the Balkans using ONLY the excerpts provided, which come from Wikivoyage travel guides.
Rules:
- Base every statement on the excerpts. Do not use outside knowledge.
- After each statement, cite the excerpt id(s) in square brackets, e.g. [wv-12].
- If the excerpts do not contain the answer, reply exactly "{REFUSAL}" and nothing else.
- Answer in 2-4 sentences of plain English."""


def load_chunks(path: Path = CHUNKS_FILE) -> pd.DataFrame:
    """Load the chunks saved by notebook 01, one row per chunk with its metadata."""
    return pd.read_json(path, lines=True)
 
 
def load_embedder(model_name: str = EMBEDDING_MODEL):
    """Load the sentence-transformers embedding model (downloaded once, then cached)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)
 
 
def make_collection(chunks: pd.DataFrame, embeddings, name: str = COLLECTION_NAME, path: Path = CHROMA_DIR):
    """Store the chunks and their embeddings in a Chroma collection saved in models/chroma/.
 
    PersistentClient instead of Client(), so the index is saved on disk and
    notebook 03 and the app can open it without embedding again. An old
    collection with the same name is replaced.
    """
    import chromadb
    chroma = chromadb.PersistentClient(path=str(path))
    if name in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(name)
    collection = chroma.create_collection(name, configuration={"hnsw": {"space": "cosine"}})
    metadatas = [{"title": c.title, "country": c.country or "", "section": c.section,
                  "url": c.url, "is_nature": bool(c.is_nature)} for c in chunks.itertuples()]
    batch = 1000                                   # Chroma limits how many rows one add() takes
    for start in range(0, len(chunks), batch):
        collection.add(
            ids=chunks["chunk_id"].iloc[start:start + batch].tolist(),
            documents=chunks["text"].iloc[start:start + batch].tolist(),
            embeddings=embeddings[start:start + batch],
            metadatas=metadatas[start:start + batch],
        )
    return collection
 
 
def open_collection(name: str = COLLECTION_NAME, path: Path = CHROMA_DIR):
    """Open the collection saved by make_collection()."""
    import chromadb
    return chromadb.PersistentClient(path=str(path)).get_collection(name)
 
 
def retrieve(question: str, collection, embedder, k: int = TOP_K, where: dict | None = WHERE) -> pd.DataFrame:
    """Embed the question with the same model, query the collection, and return the top k chunks.
 
    Columns: id, title, country, section, distance (smaller is closer), text, url.
    """
    query_embedding = embedder.encode([question], normalize_embeddings=True)
    results = collection.query(query_embeddings=query_embedding, n_results=k, where=where)
    metadatas = results["metadatas"][0]
    return pd.DataFrame({
        "id": results["ids"][0],
        "title": [m["title"] for m in metadatas],
        "country": [m["country"] for m in metadatas],
        "section": [m["section"] for m in metadatas],
        "distance": np.round(results["distances"][0], 3),
        "text": results["documents"][0],
        "url": [m["url"] for m in metadatas],
    })
 
 
def build_context(hits: pd.DataFrame) -> str:
    """Label each retrieved chunk with its id, article and section, followed by its text."""
    return "\n\n".join(f"[{row.id}] ({row.title}, {row.section})\n{row.text}" for row in hits.itertuples())
 
 
def answer(client, question: str, collection, embedder, k: int = TOP_K,
           where: dict | None = WHERE, step: str = "answers") -> dict:
    """Retrieve the chunks, ask the model to answer only from them, and return what it cited.
 
    The call is tracked under `step`. Returns the answer, the cited ids (found
    in the answer with a regular expression), the retrieved ids, and the
    titles and sections of the retrieved chunks.
    """
    hits = retrieve(question, collection, embedder, k=k, where=where)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Excerpts:\n{build_context(hits)}\n\nQuestion: {question}"},
    ]
    response = track(client.chat.completions.create(model=LLM_MODEL, messages=messages), step)
    text = response.choices[0].message.content
    return {
        "answer": text,
        "cited": sorted(set(re.findall(r"wv-\d+", text))),
        "retrieved": list(hits["id"]),
        "titles": list(hits["title"]),
        "sections": list(hits["section"]),
    }
 
 
def hit_rate(questions: pd.DataFrame, collection, embedder, k: int = TOP_K, where: dict | None = WHERE):
    """Share of questions for which at least one of the top k chunks comes from the expected article.
 
    `questions` needs `question` and `expected_title` (several titles separated
    by ";" for comparison questions: any of them counts). Returns the rate and
    the questions missed.
    """
    queries = embedder.encode(list(questions["question"]), normalize_embeddings=True)
    result = collection.query(query_embeddings=queries, n_results=k, where=where)
    hits = [any(t.strip() in [m["title"] for m in metas] for t in expected.split(";"))
            for expected, metas in zip(questions["expected_title"], result["metadatas"])]
    missed = [q for q, hit in zip(questions["question"], hits) if not hit]
    return float(np.mean(hits)), missed
 



# --- The travel log: the places user has visited, with their ratings -------------------
 
VISITED_FILE = ROOT / "data" / "visited_places.csv"    
VISITED_COLUMNS = ["title", "country", "rating", "visited_on", "note"]
AGENT_STEP_LIMIT = 10        
 
 
def load_visited(path: Path = VISITED_FILE) -> pd.DataFrame:
    """Return the travel log, one row per place visited and an empty table if there is none yet."""
    if not path.exists():
        return pd.DataFrame(columns=VISITED_COLUMNS)
    return pd.read_csv(path).fillna("")
 
 
def add_visit(title: str, rating: int, chunks: pd.DataFrame, note: str = "", visited_on: str = "",
              path: Path = VISITED_FILE) -> pd.DataFrame:
    """Add a place to the travel log, or update it if it is already there, and save the log.
 
    Called by the app's form, not by the agent: the agent's tools only read.
    The place must be an article title from the guides (so the agent can search
    it) and the rating a whole number from 1 to 5. Raises ValueError otherwise.
    """
    known = chunks.drop_duplicates("title").set_index("title")["country"]
    if title not in known.index:
        raise ValueError(f"'{title}' is not a place in the guides.")
    if int(rating) != rating or not 1 <= rating <= 5:
        raise ValueError("The rating must be a whole number from 1 to 5.")
    log = load_visited(path)
    log = log[log["title"] != title]
    new_row = pd.DataFrame([{"title": title, "country": known[title] or "", "rating": int(rating),
                             "visited_on": visited_on, "note": note}])
    log = pd.concat([log, new_row], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(path, index=False)
    return log
 
 
# --- The tools and the agent -------------------------------------------------------
 
AGENT_PROMPT = f"""You are Trail Scout, an assistant for hikers and nature travellers in the Balkans (Albania, Bosnia and Herzegovina, Bulgaria, Croatia, Kosovo, Montenegro, North Macedonia, Romania, Serbia).
Rules:
- Use search_guides for any question about places, parks, hikes, fees, getting there or staying there. Answer only from what it returns, never from your own knowledge.
- Use get_visited_places when the user asks for a recommendation or mentions the places they have been. Do not recommend a place they have already visited and use their ratings to understand what they like.
- After each statement taken from the guides, cite the excerpt id(s) in square brackets, e.g. [wv-12].
- If the guides do not contain the answer, reply exactly "{REFUSAL}" and nothing else. Live information (weather, current trail or snow conditions) and bookings are never in the guides.
- Answer in at most 5 sentences of plain English."""
 
 
def make_tools(collection, embedder, visited_path: Path = VISITED_FILE):
    """Build the agent's two read-only tools around the Chroma collection and the travel log."""
    from langchain.tools import tool
 
    @tool
    def search_guides(query: str) -> str:
        """Search the Wikivoyage travel guides for the Balkans (national parks, mountains, hikes, towns, fees, how to get there, where to sleep, safety). Use it for any question about a place."""
        hits = retrieve(query, collection, embedder, k=TOP_K, where=WHERE)
        return "\n\n".join(f"[{row.id}] ({row.title}, {row.country} - {row.section})\n{row.text}"
                           for row in hits.itertuples())
 
    @tool
    def get_visited_places() -> str:
        """List the places the user has already visited, with their rating from 1 (disliked) to 5 (loved) and their note."""
        log = load_visited(visited_path)
        if log.empty:
            return "The user has not recorded any visited places yet."
        return "\n".join(f"- {r.title} ({r.country}): rated {r.rating}/5" + (f", note: {r.note}" if r.note else "")
                         for r in log.itertuples())
 
    return [search_guides, get_visited_places]
 
 
def make_agent(collection, embedder, visited_path: Path = VISITED_FILE, memory: bool = False):
    """Create the Trail Scout agent with LangChain.
 
    With memory=True it gets an InMemorySaver checkpointer: pass a thread_id to
    run_agent() and it remembers the earlier messages of that conversation.
    """
    from langchain.agents import create_agent
    from langgraph.checkpoint.memory import InMemorySaver

    return create_agent(
        model=f"openai:{LLM_MODEL}",
        tools=make_tools(collection, embedder, visited_path),
        system_prompt=AGENT_PROMPT,
        checkpointer=InMemorySaver() if memory else None,
    )
 

def run_agent(agent, question: str, thread_id: str | None = None, step: str = "agent",
              recursion_limit: int = AGENT_STEP_LIMIT) -> dict:
    """Ask the agent a question, and return its answer with the tools it called, the chunks it saw, tokens and time.
 
    The tokens are added up in show_trace() and recorded in
    usage_log under `step`. If the step limit is reached, the answer says so.
    Returns: answer, tool_calls (name and arguments, in order), retrieved (chunk
    ids returned by search_guides), cited ids, input_tokens, output_tokens,
    seconds, stopped (True if the step limit was reached).
    """
    from langgraph.errors import GraphRecursionError

    config = {"recursion_limit": recursion_limit}
    if thread_id:
        config["configurable"] = {"thread_id": thread_id}
    start = time.perf_counter()
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": question}]}, config)
    except GraphRecursionError:
        return {"answer": "Stopped, too many steps.", "tool_calls": [], "retrieved": [], "cited": [],
                "input_tokens": 0, "output_tokens": 0, "seconds": round(time.perf_counter() - start, 2),
                "stopped": True}
    seconds = round(time.perf_counter() - start, 2)
 
    # With memory, result["messages"] holds the whole thread: count only this turn
    messages = result["messages"]
    last_question = max(i for i, m in enumerate(messages) if type(m).__name__ == "HumanMessage")
    turn = messages[last_question:]
    tokens_in = tokens_out = 0
    tool_calls, retrieved = [], []
    for message in turn:
        kind = type(message).__name__
        if kind == "AIMessage":
            tokens_in += message.usage_metadata["input_tokens"]
            tokens_out += message.usage_metadata["output_tokens"]
            tool_calls += [{"name": call["name"], "args": call["args"]} for call in message.tool_calls]
        elif kind == "ToolMessage" and message.name == "search_guides":
            retrieved += re.findall(r"\[(wv-\d+)\]", str(message.content))
    usage_log.append({"step": step, "input": tokens_in, "output": tokens_out})
    answer = turn[-1].content
    return {"answer": answer, "tool_calls": tool_calls, "retrieved": list(dict.fromkeys(retrieved)),
            "cited": sorted(set(re.findall(r"wv-\d+", answer))),
            "input_tokens": tokens_in, "output_tokens": tokens_out, "seconds": seconds, "stopped": False}
 
 
 
def show_trace(run: dict) -> None:
    """Print what run_agent() did: each tool call, the tokens, and the answer."""
    for call in run["tool_calls"]:
        print(f"  -> {call['name']}({call['args']})")
    print(f"  ({run['input_tokens']:,} input and {run['output_tokens']:,} output tokens)")
    print("\nAnswer:", run["answer"])


# --- Running the evaluation -------------------------------------------------------
 
AGENT_ANSWERS_FILE = ROOT / "data" / "agent_answers.csv"        # the evaluated answers: commit it
TOOL_TESTS_FILE = ROOT / "data" / "agent_answers_tools.csv"     # the tool-choice tests: commit it
BASELINE_LABELS_FILE = ROOT / "data" / "baseline_labels.csv"    # saved at the end of notebook 01
 
 
def run_agent_eval(agent, questions: pd.DataFrame, path: Path = AGENT_ANSWERS_FILE,
                   rerun: bool = False) -> pd.DataFrame:
    """Run the agent on every question once and save the results, or load the saved ones.
 
    As with the baseline, the answers are saved and reused, so the evaluation is
    run once and the labels keep matching the answers they were given for.
    Set rerun=True only to run it again (then label again).
    One row per question: the answer, the tools called, the chunks retrieved and
    cited (ids separated by ";"), tokens, cost and seconds.
    """
    if path.exists() and not rerun:
        return pd.read_csv(path).fillna("")
    rows = []
    for q in questions.itertuples(index=False):
        run = run_agent(agent, q.question, step=f"eval {q.question_id}")
        rows.append({
            "question_id": q.question_id,
            "answer": run["answer"],
            "tools": ";".join(call["name"] for call in run["tool_calls"]),
            "retrieved": ";".join(run["retrieved"]),
            "cited": ";".join(run["cited"]),
            "input_tokens": run["input_tokens"],
            "output_tokens": run["output_tokens"],
            "cost_usd": cost(run["input_tokens"], run["output_tokens"]),
            "seconds": run["seconds"],
            "stopped": run["stopped"],
        })
        print(f"{q.question_id}: {rows[-1]['tools'] or 'no tool'}, {run['seconds']} s")
    results = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(path, index=False)
    return results.fillna("")
 
 
def check_answers(results: pd.DataFrame, chunks: pd.DataFrame) -> pd.DataFrame:
    """Add the automatic checks to the evaluation results, one True/False column each (as in the RAG lab).
 
    `results` needs the evaluation set's columns (kind, expected_title) and
    run_agent_eval()'s (answer, retrieved, cited). For answerable questions:
    right article retrieved, cites, cites only what it saw, cites the right
    article. For all questions: refused (the exact refusal sentence).
    `failure_source` says where a failed answer went wrong, using the hand
    `label`: "retrieval" if the expected article was never retrieved,
    "generation" if it was (or if an out-of-scope question was answered).
    """
    title_of = dict(zip(chunks["chunk_id"], chunks["title"]))
    ids = lambda cell: [i for i in str(cell).split(";") if i]
    expected = lambda cell: [t.strip() for t in str(cell).split(";") if t.strip()]
    checked = results.copy()
    checked["retrieved_titles"] = checked["retrieved"].map(lambda c: sorted({title_of.get(i, "?") for i in ids(c)}))
    checked["cited_titles"] = checked["cited"].map(lambda c: sorted({title_of.get(i, "?") for i in ids(c)}))
    checked["right article retrieved"] = [any(t in got for t in expected(exp))
                                          for exp, got in zip(checked["expected_title"], checked["retrieved_titles"])]
    checked["cites"] = checked["cited"].map(lambda c: len(ids(c)) > 0)
    checked["cites only what it saw"] = [set(ids(c)) <= set(ids(r)) for c, r in zip(checked["cited"], checked["retrieved"])]
    checked["cites the right article"] = [any(t in got for t in expected(exp))
                                          for exp, got in zip(checked["expected_title"], checked["cited_titles"])]
    checked["refused"] = checked["answer"].str.contains(REFUSAL, regex=False)
    if "label" in checked:
        def source(row):
            if row["kind"] == "out_of_scope":
                return "generation" if row["label"] != "refused" else ""
            if row["label"] == "correct":
                return ""
            return "generation" if row["right article retrieved"] else "retrieval"
        checked["failure_source"] = checked.apply(source, axis=1)
    return checked