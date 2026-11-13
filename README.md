![Ironhack logo](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Project 3 | AI Product: Mountain Goat 🐐

*Data Science & Machine Learning — Week 9, the final project*

**Mountain Goat** is an AI agent for hikers and nature travellers exploring the mountains and national parks of the Balkans. It answers questions from the Wikivoyage travel guides, always with the source, says *"I don't know"* when the guides don't cover a question, and recommends where to go next based on the places you have visited and how you rated them.

**Track:** LLM (a RAG pipeline used as the main tool of an agent) · **App:** https://mountain-goat-agent.streamlit.app

`Python` · `NLP` · `Transformers` · `Embeddings` · `LLM APIs` · `RAG` · `Agents` · `Evaluation` · `Streamlit` · `MLOps`



## 1. Project Overview

**The problem:** an LLM on its own answers travel questions fluently, but it invents details (fees, routes, huts) and cannot say where its answers come from. In the mountains, **a confident wrong answer is the costliest mistake**, because a hiker acts on it. Mountain Goat answers only from the guides, cites them, and refuses when they don't cover the question.

The project follows these steps:
**Frame** (user, metric) → **Data** (download, clean, chunk) → **Evaluation set** (written before building) → **Baseline** (the LLM alone) → **RAG pipeline** → **Design choices** (chunk size, `k`, filter) → **Agent** (tools, memory, step limit) → **Evaluate once** → **App**.



## 2. The Task

| | |
|---|---|
| **Who uses it** | A hiker or nature traveller planning a trip to the Balkans, at the planning stage |
| **Input** | A question in plain English, plus their list of visited places with ratings |
| **Output** | A short answer citing the Wikivoyage excerpts it used, or *"I don't know based on my guides"* |
| **Decision it supports** | Where to go, and how to prepare: getting there, costs, where to sleep, what to watch out for |
| **Main metrics** | **Grounded accuracy** (correct *and* citing the right article), **refusals** on out-of-scope questions, **retrieval hit rate** |
| **Baseline** | The same LLM with no guides and no tools: 36% right overall, 0% grounded |

**Why these metrics:** the three possible mistakes have very different costs. A confident wrong answer (an invented trail, a wrong fee) is the costliest, because the user acts on it. An answer without a source is nearly as bad, because the user cannot tell it from a wrong one. A wrong refusal costs least: the user looks elsewhere. So answers must be correct **and** checkable, and questions the guides cannot answer must be refused. Retrieval is measured separately, because the agent can only answer from what the search finds.



## 3. The Data

- **Source:** English [Wikivoyage](https://en.wikivoyage.org), downloaded through its public MediaWiki API with `python wikivoyage_balkans.py` (categories, park lists and destination links followed, every article checked against its location in Wikivoyage's region tree)
- **Licence:** [CC BY-SA](https://creativecommons.org/licenses/by-sa/4.0/): travel guide text by Wikivoyage contributors
- **Scope:** **618 articles** for Albania, Bosnia and Herzegovina, Bulgaria, Croatia, Kosovo, Montenegro, North Macedonia, Romania and Serbia, downloaded on 6 November 2026. 
- **Files:** `data/raw/wikivoyage_balkans.jsonl` (the articles) and `wikivoyage_balkans_manifest.json` (download date, licence, counts)
- **Data quality**, found and handled in notebook 01:
  - The text is wiki markup: listings (sights, hotels), maps, banners and unit templates. `clean_wikitext()` turns listings into readable lines with their price, hours and *last checked* date, keeps units (`{{eur|5}}` → €5) and removes the rest.
  - **78% of articles are "outline" status** and 36% of sections have fewer than 15 words: coverage is uneven, so some questions have no answer in the data.
  - **41% of listings were last checked in 2020 or earlier:** the dates are kept in the text, so answers can show how old a price is.
  - **Surprising types:** Pirin National Park is tagged as a city, Matka Canyon as a district. Which articles count as *nature* is therefore decided by type **or** content (a park or rural area, or at least 1% nature words): 247 articles.
  - **Contact details:** phone numbers and emails in listings are removed or masked, so none are sent to the API.
  - **Diacritics:** 29% of titles (Žabljak); tested with a question typed without them.
- **Chunks:** each article is split by section, and long sections into **105-word chunks with an overlap of 21**: **11,510 chunks**, each with its id (`wv-12`), title, country, section, URL and nature flag. 105 words because MiniLM reads at most 256 tokens: at 150 words, 10% of chunks went over the limit.
- **Evaluation set:** `data/eval_questions.csv`, **22 questions** written before building: 18 the guides can answer (facts, practical, recommendations, safety, comparisons), across all 9 countries, each with the expected article, section and answer; and 4 they cannot (live snow conditions, a place in Slovenia, tomorrow's weather, a booking).



## 4. Pipeline Design

**1. Retrieval (RAG):** each chunk is embedded with **sentence-transformers `all-MiniLM-L6-v2`** and stored in a **Chroma** collection (cosine distance), saved in `models/chroma/`. A question is embedded with the same model, and the **5 closest chunks** are retrieved.

**2. The agent (LangChain `create_agent`, `gpt-5.4-nano`)** with two **read-only** tools:

| Tool | What it does |
|---|---|
| `search_guides(query)` | the RAG search: returns the 5 closest chunks with their ids, so the agent can cite them |
| `get_visited_places()` | reads the user's travel log: places visited, rating 1–5, note |

The system prompt tells the agent to answer only from the tools, cite the excerpt ids (`[wv-12]`), never recommend a place already visited, and reply *"I don't know based on my guides."* otherwise. It has **memory** (a checkpointer and a `thread_id` per conversation) and a **step limit** of 10.

**Read-only by design:** a tool that writes turns a wrong answer into a wrong action, so the agent never changes the travel log. The user adds and removes places through the app's form (`add_visit()`, `remove_visit()`), which only accepts places from the guides and ratings from 1 to 5.

**Design choices, compared on the evaluation set** (notebook 02):

| Choice | Compared | Result | Kept |
|---|---|---|---|
| Chunk size | 100–150 words | 150 words: 10% of chunks over MiniLM's 256 tokens; 105 words: none (or almost none) | **105 words**, overlap 21 |
| `k` and filter | k = 3 / 5, all articles / nature only | hit rate 83% (k = 3, all), 89% for the other three | **k = 5, no filter**: the filter adds nothing at k = 5 and would hide town articles |



## 5. Repository Structure

```
project-3-ai-product/
│
├── app/
│   ├── streamlit_app.py                     # the app: Home, the agent (chat and My places), Gallery, About
│   └── gallery/jpgs.csv                     # gallery photos: Wikimedia Commons URLs with their credits
│
├── data/
│   ├── raw/                                 # downloaded by download_data.py (gitignored)
│   │   ├── .gitkeep                         # keeps the empty folder in git
│   │   ├── wikivoyage_balkans.jsonl         # the 619 articles, raw wikitext
│   │   └── wikivoyage_balkans_manifest.json # download date, licence, counts
│   ├── processed/
│   │   ├── chunks.jsonl                     # the 11,510 chunks (committed for deployment; CC BY-SA)
│   │   └── demo_visited.csv                 # the demo travel log used in notebooks 02 and 03
│   ├── eval_questions.csv                   # the 22 evaluation questions
│   ├── baseline_llm_answers.csv             # the LLM-alone answers, saved once
│   ├── baseline_labels.csv                  # their labels
│   ├── agent_answers.csv                    # the agent's answers to the 22 questions, saved once
│   ├── agent_answers_tools.csv              # the agent's answers to the 4 tool-choice tests
│   └── visited_places.csv                   # your travel log, written by the app (gitignored)
│
├── models/chroma/                           # the Chroma index (gitignored, rebuilt by notebook 02 or the app)
│
├── notebooks/
│   ├── 01_framing_and_baseline.ipynb        # framing, data quality, EDA, chunks, evaluation set, baseline
│   ├── 02_build.ipynb                       # the RAG pipeline, design choices, the agent
│   └── 03_evaluate.ipynb                    # the report: evaluation, failure analysis, cost, conclusions
│
├── src/functions.py                         # all reusable logic, imported by the notebooks and the app
├── wikivoyage_balkans.py                    # fetches the Wikivoyage articles into data/raw/
├── .streamlit/config.toml                   # the app's theme
├── .env                                     # the key needed
├── requirements.txt
├── RUBRIC.md
└── README.md
```

**Main functions in `src/functions.py`:**

| Part | Functions |
|---|---|
| Loading and cleaning | `load_articles()`, `split_sections()`, `clean_wikitext()`, `nature_terms_per_1k()`, `build_chunks()`, `save_chunks()` |
| Baseline | `load_eval_set()`, `cost()`, `track()`, `ask_llm_alone()`, `run_llm_baseline()`, `labels_summary()` |
| RAG | `load_embedder()`, `make_collection()`, `open_collection()`, `retrieve()`, `build_context()`, `answer()`, `hit_rate()` |
| Agent | `make_tools()`, `make_agent()`, `run_agent()`, `show_trace()`, `load_visited()`, `add_visit()`, `remove_visit()` |
| Evaluation | `run_agent_eval()`, `check_answers()` |



## 6. Results

**The agent against the baseline, on the same 22 questions** (notebook 03, each answer labelled by hand):

| | LLM alone (baseline) | **Mountain Goat** |
|---|---|---|
| Correct on the 18 answerable questions | 28% (5) | **67% (12)** |
| Correct **and** citing the right article (grounded) | 0% | **67% (12)** |
| Out-of-scope questions refused | 75% (3 of 4) | **100% (4 of 4)** |
| **Right overall** (all 22) | 36% (8) | **73% (16)** |

- **Retrieval:** the expected article is in the top 5 for **89%** (16 of 18) of the answerable questions.
- **By kind:** facts 6 of 7 correct, practical questions 3 of 4, safety 1 of 2, comparisons 1 of 2, recommendations 1 of 3 (the other 2 refused).
- **Tool choice:** 4 of 4 tool-choice tests used the right tools (e.g. reading the travel log before recommending).

**Where it goes wrong** (6 failures, each classified):

| Type | Question | What happened |
|---|---|---|
| Retrieval | Can I drive to Theth in winter? | Thethi was not retrieved; the agent gave general driving advice and **missed that the road is closed from November to April**. The most serious failure. |
| Retrieval | Via ferrata in Kosovo | The agent's own search query found Kosovo's towns, not the Rugova Mountains; it refused. |
| Retrieval | A multi-day trek in the Accursed Mountains | The English name; the article is *Prokletije*. It refused. |
| Generation | Plitvice adult ticket in July | It picked the wrong ticket type from a section with many prices. |
| Generation | Hiking off-trail around Valbona | The right article, likely not the *Stay safe* section; it refused. |
| Generation | Swimming at Plitvice or Krka | Answered for Krka, refused for Plitvice. |

**Cost and speed:** about $0.00047 and 2.4-4.5 sec per question, about twice the cost of a plain RAG answer, because the agent makes several model calls. 1,000 questions a day would cost about $14.04 a month.

**Verdict:** worth shipping as a **research aid** that always shows its sources, **not** as a safety or route-planning tool.



## 7. Limitations and Next Steps

**It should not be trusted for:**
- **Live information:** weather, snow and trail conditions, opening today.
- **Exact prices:** sections with several ticket types confuse it, and many prices are years old (check the *last checked* dates).
- **Places outside the nine countries** (Slovenia, Greece) and **names the guides do not use** (Accursed Mountains vs Prokletije).
- **Safety decisions and route planning:** it can miss a critical detail, as with the road to Theth.

**Other limits:** the design choices were made on the evaluation set, so its retrieval scores are slightly optimistic; 22 questions is small (one question ~ 6 points); the travel log is a single CSV file, shared by everyone on a public deployment and reset when the app restarts.

**With more time:**
- add English and alternative place names to the chunks (Accursed Mountains → Prokletije);
- improve retrieval at section level, so *Stay safe* or *Fees and permits* is found for safety and price questions;
- a larger evaluation set, with more recommendation questions;
- one travel log per user (a database);
- expand to more of Europe, so someone who loved the Alps can discover similar, less crowded places.



## 8. The App

`app/streamlit_app.py` has four pages, chosen in the sidebar:

- **🏔️ Home:** what Mountain Goat is.
- **ℹ️ About:** the data, how it works, the evaluation and the limits.
- **🐐 Mountain Goat:** the chat (with memory, the cited sources as expanders with links to Wikivoyage, a clear message when the guides don't cover a question, and the tools, tokens, cost and time of each answer), and **My places**: a form to add or remove places with a rating and a note, and the travel log.
- **📸 Gallery:** photos of places in the guides, with links to their articles and their credits.

The embedding model, the index and the agent are loaded **once** with `st.cache_resource`. If no saved index exists (e.g. on Streamlit Cloud), the app builds it from `data/processed/chunks.jsonl` on its first start, which takes a few minutes.



## 9. How to Run

Built with Python [TODO: your version, e.g. 3.12].

```bash
# 1. Clone the repository
git clone https://github.com/jkrstevska/project-3-ai-product/tree/main

# 2. Create a virtual environment and install the libraries
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Add your OpenAI key (never commit .env)
cp .env.example .env                 

# 4. Download the Wikivoyage articles into data/raw/
python download_data.py

# 5. Run the notebooks in order: 01 → 02 → 03
#    01 saves the chunks, 02 builds the index in models/chroma/, 03 is the report
jupyter lab notebooks/

# 6. Run the app
streamlit run app/streamlit_app.py
```

The saved answers in `data/` mean the evaluations are not re-run (and not paid for again) when the notebooks are re-run. To deploy on Streamlit Community Cloud, add `OPENAI_API_KEY = "sk-..."` under the app's **Settings → Secrets**.



## 10. Credits and Use of AI

- **Travel guide text:** Wikivoyage (en.wikivoyage.org), by Wikivoyage contributors, licensed CC BY-SA. `data/processed/chunks.jsonl` is derived from it and committed under the same licence.
- **Photos:** Wikimedia Commons; each photo's author and licence are in `app/gallery/jpgs.csv` and shown in the app.
- **Use of AI:** I used Claude (Anthropic) as a coding assistant throughout the project. It wrote the first version of `wikivoyage_balkans.py`, most of the functions in `src/functions.py`, the app, and drafts of notebook cells, findings and the evaluation questions. I chose the project, the scope and the data; checked every output against the data; aligned the code with the methods taught in class (the RAG, Agents and Streamlit labs); made the design decisions (105-word chunks, k = 5, read-only agent tools); verified the evaluation answers in the articles; labelled every answer; and wrote and edited the final text.



## 11. Author

Jovana Krstevska