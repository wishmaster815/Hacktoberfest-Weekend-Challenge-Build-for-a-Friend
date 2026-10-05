"""
Recipe Keeper - Streamlit UI (runs locally with Ollama + faster-whisper)

Run:
    pip install -r requirements.txt
    ollama pull llama3.1:8b
    streamlit run app.py

Keep this file in the same folder as recipe_book.py.
"""
import json
import tempfile
import urllib.request
from pathlib import Path

import streamlit as st

from recipe_book import PROMPT, ask_llm, render

st.set_page_config(page_title="Recipe Keeper", page_icon="🍲", layout="centered")


@st.cache_resource(show_spinner="Loading speech model (first time downloads it)...")
def load_whisper(size):
    from faster_whisper import WhisperModel

    return WhisperModel(size, compute_type="int8")


def transcribe(data, suffix, size, translate):
    model = load_whisper(size)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
        path = f.name
    try:
        task = "translate" if translate else "transcribe"
        segments, info = model.transcribe(path, task=task)
        return " ".join(s.text.strip() for s in segments), info.language
    finally:
        Path(path).unlink(missing_ok=True)


def ollama_models():
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as r:
            return [m["name"] for m in json.loads(r.read())["models"]]
    except Exception:
        return None


# ---------- sidebar ----------
st.sidebar.header("Settings")
cook = st.sidebar.text_input("Who is telling the recipes?", value="Grandmother")
models = ollama_models()
if models:
    model = st.sidebar.selectbox("Ollama model", models)
else:
    model = None
whisper_size = st.sidebar.selectbox("Whisper size", ["tiny", "base", "small", "medium"], index=2)
translate = st.sidebar.checkbox("Translate speech to English", value=False)

# ---------- main ----------
st.title("🍲 Recipe Keeper")
st.caption("Voice memos in, family recipe book out. Everything runs on this computer.")

if not models:
    st.error(
        "Can't reach Ollama. Start it (open the Ollama app or run `ollama serve`) "
        "and pull a model, e.g. `ollama pull llama3.1:8b`, then refresh."
    )
    st.stop()

if "recipes" not in st.session_state:
    st.session_state.recipes = []

audio_files = st.file_uploader(
    "Upload voice memos",
    type=["mp3", "m4a", "wav", "ogg", "flac", "aac", "opus"],
    accept_multiple_files=True,
)
pasted = st.text_area("...or paste a transcript", height=120, placeholder="Okay so for dal, I take one cup...")

if st.button("Write it down", type="primary", disabled=not (audio_files or pasted.strip())):
    jobs = []
    for f in audio_files or []:
        jobs.append((f.name, f.getvalue(), Path(f.name).suffix))
    if pasted.strip():
        jobs.append(("pasted text", None, None))

    progress = st.progress(0.0)
    for n, (name, data, suffix) in enumerate(jobs, 1):
        with st.status(f"{name}", expanded=False) as status:
            if data is None:
                transcript, lang = pasted.strip(), "n/a"
            else:
                status.update(label=f"{name}: transcribing...")
                transcript, lang = transcribe(data, suffix, whisper_size, translate)
            status.update(label=f"{name}: writing recipe...")
            try:
                recipe = ask_llm(PROMPT.format(cook=cook, transcript=transcript), model)
            except Exception as e:
                st.warning(f"LLM failed ({e}). Saved the raw transcript instead.")
                recipe = {"title": Path(name).stem, "story": "", "ingredients": [], "steps": [], "tips": []}
            st.session_state.recipes.append(render(recipe, transcript, cook, lang))
            status.update(label=f"{name}: done", state="complete")
        progress.progress(n / len(jobs))

if st.session_state.recipes:
    st.divider()
    st.subheader(f"{cook}'s Recipes")
    for md in st.session_state.recipes:
        st.markdown(md, unsafe_allow_html=True)

    book = f"# {cook}'s Recipes\n\n---\n\n" + "\n".join(st.session_state.recipes)
    c1, c2 = st.columns(2)
    c1.download_button("Download recipe book (.md)", book, file_name="recipe_book.md", mime="text/markdown")
    if c2.button("Clear all"):
        st.session_state.recipes = []
        st.rerun()
