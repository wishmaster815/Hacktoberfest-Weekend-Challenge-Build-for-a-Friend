#!/usr/bin/env python3
"""
Voice memos -> family recipe book. 100% local, open-source.

Pipeline:
  audio (.mp3/.m4a/.wav/.ogg)  --faster-whisper-->  transcript
  transcript                   --Ollama (open-weight LLM)--> structured recipe
  recipes                      --> one markdown book

Setup:
  pip install faster-whisper
  # install Ollama from https://ollama.com, then:
  ollama pull llama3.1:8b        # or qwen2.5:7b, gemma2:9b, mistral ...

Usage:
  python recipe_book.py memos/ --cook "Nani" --out nani_recipes.md
  python recipe_book.py memos/ --cook "Dadaji" --translate      # Hindi etc. -> English
  python recipe_book.py memos/ --model qwen2.5:7b --whisper medium

Put .txt files in the folder to skip transcription (handy for testing).
"""
import argparse
import json
import urllib.request
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
AUDIO_EXT = {".mp3", ".m4a", ".wav", ".ogg", ".flac", ".aac", ".opus"}

PROMPT = """You turn a spoken recipe into a clean written one.
The speaker is {cook}, talking casually, so expect rambling, vague amounts
("a handful", "until it smells right") and side stories.

Rules:
- Do NOT invent ingredients or steps that were not said.
- Keep vague amounts in {cook}'s own words (e.g. "a fistful of coriander").
  If no amount was given, leave it empty.
- Keep any personal story or memory in "story", lightly cleaned up.
- Keep tips and warnings ("don't let the onions burn!") in "tips".

Return ONLY JSON with this shape:
{{"title": str, "story": str, "ingredients": [{{"item": str, "amount": str}}],
  "steps": [str], "tips": [str]}}

Transcript:
\"\"\"{transcript}\"\"\"
"""


def transcribe(path, model_size, translate):
    from faster_whisper import WhisperModel  # lazy import: only needed for audio

    model = WhisperModel(model_size, compute_type="int8")  # runs fine on CPU
    task = "translate" if translate else "transcribe"
    segments, info = model.transcribe(str(path), task=task)
    text = " ".join(s.text.strip() for s in segments)
    return text, info.language


def ask_llm(prompt, model):
    body = json.dumps(
        {"model": model, "prompt": prompt, "stream": False, "format": "json",
         "options": {"temperature": 0.2}}
    ).encode()
    req = urllib.request.Request(OLLAMA_URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(json.loads(r.read())["response"])


def render(recipe, transcript, cook, lang):
    lines = [f"## {recipe.get('title') or 'Untitled recipe'}", f"*as told by {cook}*", ""]
    if recipe.get("story"):
        lines += [f"> {recipe['story']}", ""]
    if recipe.get("ingredients"):
        lines.append("### Ingredients")
        for i in recipe["ingredients"]:
            amt = f"{i.get('amount', '').strip()} " if i.get("amount") else ""
            lines.append(f"- {amt}{i.get('item', '')}")
        lines.append("")
    if recipe.get("steps"):
        lines.append("### Method")
        lines += [f"{n}. {s}" for n, s in enumerate(recipe["steps"], 1)]
        lines.append("")
    if recipe.get("tips"):
        lines.append("### Tips")
        lines += [f"- {t}" for t in recipe["tips"]]
        lines.append("")
    lines += [
        "<details><summary>In their own words (original transcript)</summary>",
        "",
        transcript,
        "",
        "</details>",
        "",
        "---",
        "",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("folder", type=Path, help="folder of voice memos (or .txt transcripts)")
    ap.add_argument("--cook", default="Grandma", help="who is telling the recipes")
    ap.add_argument("--model", default="llama3.1:8b", help="Ollama model name")
    ap.add_argument("--whisper", default="small", help="whisper size: tiny/base/small/medium/large-v3")
    ap.add_argument("--translate", action="store_true", help="translate speech to English")
    ap.add_argument("--out", type=Path, default=Path("recipe_book.md"))
    args = ap.parse_args()

    files = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in AUDIO_EXT | {".txt"})
    if not files:
        raise SystemExit(f"No audio or .txt files found in {args.folder}")

    book = [f"# {args.cook}'s Recipes", "", f"*{len(files)} recipes, written down from voice memos.*", "", "---", ""]
    for n, f in enumerate(files, 1):
        print(f"[{n}/{len(files)}] {f.name}")
        if f.suffix.lower() == ".txt":
            transcript, lang = f.read_text(encoding="utf-8").strip(), "n/a"
        else:
            print("  transcribing...")
            transcript, lang = transcribe(f, args.whisper, args.translate)
        print("  structuring recipe...")
        try:
            recipe = ask_llm(PROMPT.format(cook=args.cook, transcript=transcript), args.model)
        except Exception as e:  # keep going; never lose the transcript
            print(f"  LLM failed ({e}); saving raw transcript instead")
            recipe = {"title": f.stem, "story": "", "ingredients": [], "steps": [], "tips": []}
        book.append(render(recipe, transcript, args.cook, lang))

    args.out.write_text("\n".join(book), encoding="utf-8")
    print(f"\nDone -> {args.out}")


if __name__ == "__main__":
    main()
