"""The app you demo on Friday. Run it from the repo root:

    streamlit run app/streamlit_app.py

This is a skeleton: replace every TODO with your own product. Day 3 of Week 8
(Streamlit, from notebook to app) shows each piece.
"""
import os
import sys
import uuid

import streamlit as st

sys.path.append(".")                       # so `src` is importable when you run from the repo root
from src.functions import *                # noqa: E402,F401  the same functions as your notebooks


def api_key():
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        from dotenv import load_dotenv
        load_dotenv()
        return os.environ.get("OPENAI_API_KEY")


@st.cache_data
def load_guides():
    """The chunks, to show the cites sources and to list the places in the guides."""
    return load_chunks()


@st.cache_resource(show_spinner='Loading the guides (once)...')
def load_core():
    """The embedding model, saved index and the agent, ONCE per session (not on every rerun).

    LLM track:   return the Chroma collection and anything else the pipeline needs.
                 Read the key with st.secrets["OPENAI_API_KEY"] or from .env, never from the code.
    """
    embedder = load_embedder()
    collection = open_collection()
    agent = make_agent(collection, embedder, VISITED_FILE, memory=True)
    return embedder, collection, agent


key = api_key()
if not key:
    st.error("No OpenAI key found. Add it to .streamlit/secrets.toml or .env (see the README).")
    st.stop()


try:
    chunks = load_guides()
    embedder, collection, agent = load_core()
except Exception as error:
    st.error("The guides are not ready yet. Run 'wikivoyage_balkans.py', then notebook 01 and 02" \
    f"which save the chunks and the index.\n\nDetails: {error}")
    st.stop()

by_id = chunks.set_index('chunk_id')
places = sorted(chunks['title'].unique())

if 'messages' not in st.session_state:
    st.session_state.messages=[]
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


def show_sources(cited, retrieved):
    """The excerpts the agent read, as expanders; the ones it cited are marked ✅ and listed first."""
    if not retrieved:
        return
    if not cited:
        return
    st.markdown('**Sources**')
    for chunk_id in cited:
    # sorted(retrieved, key=lambda i: i not in cited):
        if chunk_id not in by_id.index:
            continue
        chunk = by_id.loc[chunk_id]
        mark = '✅' if chunk_id in cited else ""
        with st.expander(f"{mark}{chunk['title']} ({chunk['country']}) - {chunk['section']}"):
            st.write(chunk['text'])
            st.markdown(f"[Read the full article on Wikivoyage]({chunk['url']})")


with st.sidebar:
    st.title(f"Explore")
    page = st.radio("Go to", ["🏔️  Home", "ℹ️  About", "🐐  Chat Agent", "📸  Gallery"], label_visibility="collapsed")


if page == '🏔️  Home':
    st.set_page_config(page_title="Home", page_icon="🏔️", layout='wide')
    st.title("🏔️ Home")
    st.write("Welcome to the Home page of the Mountain Goat agent. On the left hand-side, under the Explore section, you will find the About, Chat Agent, and Gallery"
    " sections. You can easily navigate through each section where you can learn more about the app, how the agent works, additional functionalities, and look at the gallery.\n\n "
    "Why is it called Mountain Goat?\n\n"
    "Because no one knows the mountains better.")
    left, right = st.columns(2)
    with left:
        st.image("https://upload.wikimedia.org/wikipedia/commons/1/1c/Hangerer_-_goats_2.jpg?utm_source=commons.wikimedia.org&utm_campaign=index&utm_content=original",
                width=800)
        st.caption("Tiia Monto, CC BY-SA 3.0, via Wikimedia Commons")
    with right:
        st.image('https://upload.wikimedia.org/wikipedia/commons/6/6f/Hangerer_-_goats_3.jpg?utm_source=commons.wikimedia.org&utm_campaign=index&utm_content=original',
                width=535)
        st.caption("Tiia Monto, CC BY-SA 3.0, via Wikimedia Commons")
    


if page == 'ℹ️  About':
    st.set_page_config(page_title="About Mountain Goat", page_icon="ℹ️", layout='wide')
    st.title("ℹ️ About Mountain Goat")
    st.caption("What is Mountain Goat?\n\n "
    "Mountain Goat helps you discover new mountains and national parks in the Balkan Region. " \
    "It answers from the Wikivoyage travel guides, with their sources, and gives you recommendations based on places you have already visited and how you rated them.")

    st.write(f"**{len(places)} travel guides** from Wikivoyage for 9 countries:")

    # One dot per country, at its centre, sized by how many guides it has
    centres = pd.DataFrame({
        "country": ["Albania", "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Kosovo",
                    "Montenegro", "North Macedonia", "Romania", "Serbia"],
        "lat": [41.15, 44.17, 42.73, 45.10, 42.60, 42.71, 41.61, 45.94, 44.02],
        "lon": [20.17, 17.79, 25.49, 15.20, 20.90, 19.37, 21.75, 24.97, 21.00],
    })
    guides = chunks.drop_duplicates("title").groupby("country").size().rename("guides")
    centres = centres.join(guides, on="country")
    centres["size"] = centres["guides"] * 400            # dot radius in metres: more guides, bigger dot

    left, right = st.columns([2, 1])
    with left:
        st.map(centres, latitude="lat", longitude="lon", size="size", color="#4f9b63")
    with right:
        st.dataframe(centres[["country", "guides"]].sort_values("guides", ascending=False),
                    hide_index=True, width="stretch")
    st.write("Mountain Goat is an agent with two read-only tools: a search over the guides (*MiniLM* embeddings, "
            f"the {TOP_K} closest excerpts) and your list of visited places. It uses the *{LLM_MODEL}* model.")
    st.write("**Evaluation:** 22 test questions, correctly answered 73% of the time, against 36% for the same "
            "model without the guides and it declined every question the guides cannot answer.")
    st.warning("Use it to discover and prepare, not for safety decisions or route planning: "
            "it can miss important details (such as a road closed in winter). Always open the sources before acting.")
    st.caption("Travel guide text from Wikivoyage (en.wikivoyage.org), by Wikivoyage contributors, CC BY-SA. "
            "Prices may be out of date: check the \"last checked\" dates.")
    

if page == '🐐  Chat Agent':
    st.set_page_config(page_title="Mountain Goat Agent", page_icon="🐐", layout='wide')
    st.title("🐐 Mountain Goat Agent")
    st.caption("How can I help you?\n\n "
    "")

    tab_chat, tab_places = st.tabs(["🐐 Ask Mountain Goat", "📍 My places"])
    with tab_chat:
        for message in st.session_state.messages:
            with st.chat_message(message['role']):
                st.markdown(message['content'])
                if message['role'] == 'assistant':
                    show_sources(message.get('cited', []), message.get('retrieved', []))
        
        question = st.chat_input('e.g I loved Durmitor. Which national park should I visit next?')
        if question:
            with st.chat_message('user'):
                st.markdown(question)
            if len(question) > 500:
                with st.chat_message('assistant'):
                    st.warning('Please keep your message under 500 characters.')
            else:
                st.session_state.messages.append({'role': 'user', 'content': question})
                with st.chat_message('assistant'):
                    try:
                        with st.spinner("Searching the guides..."):
                            # st.write("thread:", repr(st.session_state.thread_id), "| run_agent defined at line", run_agent.__code__.co_firstlineno)
                            run = run_agent(agent, question, thread_id=st.session_state.thread_id, step="app")
                    except Exception as error:
                        st.error(f"Something went wrong while answering (the OpenAI API may be unavailable). "
                                f"Please try again.\n\nDetails: {error}")
                        st.stop()
                    if run["stopped"]:
                        st.warning("That question needed too many steps. Try asking one thing at a time.")
                    elif REFUSAL in run["answer"]:
                        st.info("The guides don't cover this so I cannot give you an answer.")
                    st.markdown(run["answer"])
                    show_sources(run["cited"], run["retrieved"])
                    tools = ", ".join(dict.fromkeys(call["name"] for call in run["tool_calls"])) or "none"
                    st.caption(f"Tools used: {tools} · {run['input_tokens']:,} input and {run['output_tokens']:,} "
                            f"output tokens (${cost(run['input_tokens'], run['output_tokens']):.4f}) · {run['seconds']} s")
                st.session_state.messages.append({"role": "assistant", "content": run["answer"],
                                                "cited": run["cited"], "retrieved": run["retrieved"]})

    with tab_places:
        st.caption('Add places you have visited. Mountain Goat reads this list when it recommends where to go next'
        ' and avoids suggesting places you have already seen.')
        with st.form('add place', clear_on_submit=True):
            place = st.selectbox('Place', places, index=None, placeholder='Choose a place from the guides or type the name')
            rating = st.slider('Your rating', min_value=1, max_value=5, value=4)
            left, right = st.columns(2)
            visited_on = left.date_input('Visited on', value=None)
            note = right.text_input('Note (optional)', placeholder='e.g. I loved the lakes')
            saved = st.form_submit_button('Save', type='primary')
        if saved:
            if place is None:
                st.error('Please choose a place first.')
            else:
                try:
                    add_visit(place, rating, chunks, note=note, visited_on=str(visited_on or ""))
                    st.success(f'Saved: {place}, {rating}/5.')
                except ValueError as error:
                    st.error(str(error))


        log = load_visited()
        if log.empty:
            st.info('No places saved yet. Add the first one above')
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric('Places visited', len(log))
            col2.metric('Countries', log['country'].nunique())
            col3.metric('Average rating', f'{log['rating'].astype(float).mean():.1f} / 5')
            st.dataframe(log.rename(columns=str.capitalize), width='stretch', hide_index=True)
    
    st.write(f"{len(st.session_state.messages)} messages in this conversation")
    if st.button("New conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()


if page == "📸  Gallery":
    st.title("📸 Gallery")
    st.write("Some of the places in the guides.")

    gallery_dir = Path("app/gallery")
    photos = pd.read_csv(gallery_dir / "jpgs.csv")
    columns = st.columns(3)
    for i, photo in enumerate(photos.itertuples()):
        with columns[i % 3]:
            st.image(photo.url, width="stretch")
            st.markdown(f"**{photo.place}**")
            if photo.place in places:                       # in the guides: link to its article
                url = chunks.loc[chunks["title"] == photo.place, "url"].iloc[0]
                st.markdown(f"[Read the guide]({url})")
            st.caption(f"Photo: {photo.credit}")

