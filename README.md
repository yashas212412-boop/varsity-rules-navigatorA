# Installs the required libraries. This takes about one minute.
!pip install -q pypdf sentence-transformers "gradio>=6,<7" google-genai


# STUDENT TODO: the Gemma 4 model that writes the answers.
#   "gemma-4-26b-a4b-it"  Mixture of Experts, 26B total / about 4B active parameters. Fast. (default)
#   "gemma-4-31b-it"      Dense, 31B parameters. More capable, slower.
GEMMA_MODEL = "gemma-4-26b-a4b-it"

# STUDENT TODO: the number of words in each chunk.
#   Small chunks (50)   more precise matches, but may lose context
#   Large chunks (350)  more context, but less precise and longer prompts
CHUNK_SIZE = 350

# STUDENT TODO: the number of words shared by neighbouring chunks, so sentences are not split.
CHUNK_OVERLAP = 50

# STUDENT TODO: the number of chunks given to Gemma for each question.
TOP_K = 5

# STUDENT TODO: the rules Gemma follows when answering. The rule to use only the context prevents invented answers.
ANSWER_RULES = """You answer questions about a document using ONLY the context passages provided.
- Each passage begins with its page number, for example [Page 2].
- After each fact you use, cite the page in brackets, for example (p. 2).
- If the answer is not in the context, reply exactly: "The answer could not be found in the document."
- Keep answers concise: 2 to 4 sentences."""

TEMPERATURE = 0.2       # a low value keeps answers consistent and close to the document

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # STUDENT TODO (optional): try "BAAI/bge-small-en-v1.5"
print("Settings configured.")


import os
from google import genai
from google.genai import types, errors


def _get_secret(name):
    """Reads a secret from the Colab Secrets panel (or from an environment variable)."""
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return os.environ.get(name)


GOOGLE_API_KEY = "AQ.Ab8RN6JvUdDeJDLzIm9P2BPyObFzj0E0iilzztpI4TrSPqjuUQ"
DEMO_MODE = not GOOGLE_API_KEY
client = None if DEMO_MODE else genai.Client(api_key=GOOGLE_API_KEY)


def _friendly_error(e):
    code = getattr(e, "code", None)
    if code in (400, 401, 403) and "key" in str(e).lower():
        return "Error: the GOOGLE_API_KEY was rejected. Check that the complete key is saved in Colab Secrets, then re-run this cell."
    if code == 429:
        return "Error: the free-tier usage limit has been reached. Wait about one minute and try again."
    if code == 404:
        return f"Error: the model '{GEMMA_MODEL}' was not found. Check the GEMMA_MODEL setting."
    return f"Error: the request to Gemma failed ({e})."


def generate(user_text, system_text=None, temperature=TEMPERATURE, demo_reply="(Demonstration mode response)"):
    """Sends one message to Gemma and returns its reply as text.
    In demonstration mode, returns `demo_reply`. If the request fails, returns a message beginning with "Error:"."""
    if DEMO_MODE:
        return demo_reply
    try:
        response = client.models.generate_content(
            model = GEMMA_MODEL,
            contents = user_text,
            config = types.GenerateContentConfig(
                system_instruction = system_text,
                temperature = temperature,
                thinking_config = types.ThinkingConfig(thinking_level="minimal"),  # faster replies
            ),
        )
        return (response.text or "").strip()
    except errors.APIError as e:
        return _friendly_error(e)
    except Exception as e:
        return f"Error: {e}"


if DEMO_MODE:
    print("Demonstration mode: no GOOGLE_API_KEY was found. Retrieval will work; answers will show the passages found.")
else:
    print(f"Connected to Google AI Studio. Model: {GEMMA_MODEL}")
    print(generate("In one sentence, what is an open-weight AI model?"))

from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer(EMBED_MODEL_NAME)   # runs on the CPU
print(f"Loaded embedding model: {EMBED_MODEL_NAME}")



# 1. Install PyMuPDF just in case
!pip install PyMuPDF -q

import glob
import fitz  # PyMuPDF

# 2. Find all PDF files in the Colab folder
pdf_files = glob.glob("*.pdf")
print(f"📁 Found these PDF files: {pdf_files}")

# 3. Read every single page from your real uploaded PDFs
SAMPLE_PAGES = []
for filename in pdf_files:
    doc = fitz.open(filename)
    print(f"📖 Reading '{filename}' ({len(doc)} pages)...")
    for page_num in range(len(doc)):
        text = doc[page_num].get_text()
        if text.strip():
            SAMPLE_PAGES.append(text)

print(f"🎉 Success! Total real pages loaded: {len(SAMPLE_PAGES)}")



# =============================================================
# 1. Install required packages (Colab)
# =============================================================
import importlib.util
import subprocess
import sys


def ensure(pip_name, module_name):
    """Install a package only if it is missing (Colab already has most of them)."""
    if importlib.util.find_spec(module_name) is None:
        print(f"Installing {pip_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])


ensure("PyMuPDF", "fitz")
ensure("google-genai", "google.genai")
ensure("sentence-transformers", "sentence_transformers")
ensure("gradio", "gradio")

import os
import glob
import fitz  # PyMuPDF
import numpy as np
import gradio as gr
from sentence_transformers import SentenceTransformer

# =============================================================
# 2. Load all PDF files from the Colab folder
# =============================================================
pdf_files = glob.glob("*.pdf")
print(f"📁 Found these PDF files: {pdf_files}")

if not pdf_files:
    raise FileNotFoundError("No PDF files found. Upload your handbooks to the Colab file panel first.")

import re


def clean_text(text):
    text = text.replace("\u00ad", "")               # soft hyphens
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)    # re-join words split across lines
    text = re.sub(r"\s*\n\s*", " ", text)           # line breaks -> spaces
    text = re.sub(r"[ \t]+", " ", text)             # repeated spaces
    return text.strip()


PAGES = []  # each item: {"file": ..., "page": ..., "text": ...}
for filename in pdf_files:
    doc = fitz.open(filename)
    empty_pages = []
    for page_index in range(len(doc)):
        # sort=True returns text in natural reading order (helps with columns)
        text = clean_text(doc[page_index].get_text("text", sort=True))
        if len(text) > 20:
            PAGES.append({"file": filename, "page": page_index + 1, "text": text})
        else:
            empty_pages.append(page_index + 1)
    print(f"📖 '{filename}': {len(doc)} pages, {len(doc) - len(empty_pages)} with readable text")
    if empty_pages:
        print(f"   ⚠️ No readable text on pages {empty_pages}. "
              "These are probably scanned images and need OCR.")
    doc.close()

if not PAGES:
    raise ValueError("No readable text found in any PDF. The files are probably scanned images.")

print(f"🎉 Success! Total pages loaded: {len(PAGES)}")
print("\n--- Sample of what was extracted (first page) ---")
print(PAGES[0]["text"][:500])
print("-------------------------------------------------\n")

# =============================================================
# 3. Embedding model, LLM helper, and RAG knowledge base
# =============================================================
embedder = SentenceTransformer("all-MiniLM-L6-v2")

CHUNK_SIZE = 100      # words per chunk
CHUNK_OVERLAP = 20    # words shared between neighbouring chunks
TOP_K = 3

# Tried in order; the first one your key can use is the one that answers.
MODEL_NAMES = ["gemma-4-26b-a4b-it", "gemma-4-31b-it"]

ANSWER_RULES = (
    "You are a campus regulations assistant. Answer ONLY using the context "
    "provided. Cite the file name and page number for each fact. If the "
    "answer is not in the context, say you could not find it in the handbooks."
)


KEY_NAMES = ("GOOGLE_API_KEY", "GEMINI_API_KEY")


def get_api_key():
    """Look for the key in environment variables, then in Colab Secrets."""
    for name in KEY_NAMES:
        key = os.environ.get(name)
        if key:
            return key
    try:
        from google.colab import userdata
    except Exception:
        return None
    for name in KEY_NAMES:
        try:
            key = userdata.get(name)
            if key:
                return key
        except Exception:
            continue  # secret missing or notebook access not enabled
    return None


def generate(prompt, system_text, demo_reply):
    """Call Gemma through the Gemini API; fall back to demo_reply if no key."""
    api_key = get_api_key()
    if not api_key:
        return demo_reply
    from google import genai
    client = genai.Client(api_key=api_key)
    last_error = None
    for model_name in MODEL_NAMES:
        try:
            # The rules are placed at the top of the prompt instead of
            # using a separate system instruction.
            response = client.models.generate_content(
                model=model_name,
                contents=f"{system_text}\n\n{prompt}",
            )
            return response.text
        except Exception as e:
            last_error = e  # try the next model name
    return f"⚠️ The model call failed: {last_error}"


knowledge_base = {"name": None, "chunks": [], "vectors": None}


def split_into_chunks(pages):
    chunks = []
    step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    for p in pages:
        words = p["text"].split()
        for start in range(0, len(words), step):
            chunk_words = words[start:start + CHUNK_SIZE]
            if len(chunk_words) > 5:
                chunks.append({
                    "file": p["file"],
                    "page": p["page"],
                    "text": " ".join(chunk_words),
                })
            if start + CHUNK_SIZE >= len(words):
                break
    return chunks


def build_knowledge_base(pages, name):
    chunks = split_into_chunks(pages)
    if not chunks:
        raise ValueError("No text was found. The PDF may contain scanned images only.")
    vectors = embedder.encode(
        [c["text"] for c in chunks],
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=True,
    )
    knowledge_base.update(name=name, chunks=chunks, vectors=vectors)
    return f"Loaded **{name}**: {len(pages)} pages, {len(chunks)} chunks."


def retrieve(question, top_k=TOP_K):
    question_vector = embedder.encode([question], normalize_embeddings=True)[0]
    similarities = knowledge_base["vectors"] @ question_vector
    best = np.argsort(similarities)[::-1][:top_k]
    return [{**knowledge_base["chunks"][i], "score": float(similarities[i])} for i in best]


def answer(question):
    if not question or not question.strip():
        return "Please type a question first.", []

    found = retrieve(question)
    context = "\n\n".join(f"[{c['file']} - page {c['page']}] {c['text']}" for c in found)
    prompt = f"Context:\n{context}\n\nQuestion: {question}"

    top = found[0]
    demo_reply = (
        "**Demonstration mode:** no API key is set, so Gemma did not write an answer. "
        f"Retrieval completed; the most relevant passage is in **{top['file']}, page {top['page']}**:\n\n"
        f"> {top['text'][:400]}..."
    )
    reply = generate(prompt, system_text=ANSWER_RULES, demo_reply=demo_reply)
    return reply, found


status_message = build_knowledge_base(PAGES, "Official College Handbooks")
print(status_message)

# =============================================================
# 4. Gradio UI with custom emerald styling
# =============================================================
custom_css = """
body {
    background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
    font-family: 'Inter', sans-serif;
}
.gradio-container {
    max-width: 750px !important;
    border-radius: 16px;
    box-shadow: 0 10px 25px -5px rgba(16, 185, 129, 0.1);
    border: 1px solid #d1fae5;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft(primary_hue="emerald")) as app:
    gr.Markdown("# 🎓 Campus Regulations Assistant")
    gr.Markdown("Your AI-powered guide to official college handbooks, rules, and policies.")
    gr.Markdown(f"✅ **Connected to your uploaded college rulebooks ({len(PAGES)} pages loaded).**")

    question_input = gr.Textbox(
        label="Question",
        placeholder="Ask about hostel rules, ragging policy, fees...",
        lines=1,
    )
    ask_btn = gr.Button("Ask", variant="primary")

    answer_output = gr.Markdown(label="Answer")
    retrieved_output = gr.JSON(label="Retrieved passages (the text given to Gemma)")

    ask_btn.click(fn=answer, inputs=question_input, outputs=[answer_output, retrieved_output])
    question_input.submit(fn=answer, inputs=question_input, outputs=[answer_output, retrieved_output])

# share=True gives you a live public link
app.launch(share=True)
print("✅ App is running. Use the public link above.")



import os
import gradio as gr

gr.close_all()  # closes any previous version of the application that is still running

APP_TITLE = "Campus Regulations Assistant"
APP_DESCRIPTION = "Your AI-powered guide to official college handbooks, rules, and policies."

def on_ask(question):
    if not question.strip():
        raise gr.Error("Enter a question first.")
    reply, found = answer(question)
    sources = "\n\n".join(f"**Page {c['page']}** (similarity {c['score']:.2f})\n> {c['text']}" for c in found)
    return reply, sources

# Custom CSS for a modern, glowing gradient background and clean cards
custom_css = """
body {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    font-family: 'Inter', sans-serif;
}
.gradio-container {
    max-width: 900px !important;
    border-radius: 16px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    background: rgba(255, 255, 255, 0.95);
    padding: 25px;
    margin-top: 30px;
    margin-bottom: 30px;
}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="emerald", secondary_hue="blue"), css=custom_css) as demo:
    gr.Markdown("# 🎓 Campus Regulations Assistant")
    gr.Markdown("### *Your AI-powered guide to official college handbooks, rules, and policies.*")
    gr.Markdown("✅ *Connected to your uploaded college rulebooks.*")

    # Upload box and sample document buttons have been completely removed here!

    question_box = gr.Textbox(label="Question", placeholder="For example: What is the minimum attendance required?")
    ask_btn = gr.Button("Ask", variant="primary")

    gr.Examples([
        "What is the minimum attendance required?",
        "Are re-evaluations allowed and what is the deadline?",
        "What are the hostel visiting hours?",
        "Who won the 2022 World Cup?"
    ], inputs=question_box)

    answer_md = gr.Markdown()
    with gr.Accordion("Retrieved passages (the text given to Gemma)", open=False):
        sources_md = gr.Markdown()

    ask_btn.click(on_ask, inputs=question_box, outputs=[answer_md, sources_md])
    question_box.submit(on_ask, inputs=question_box, outputs=[answer_md, sources_md])

demo.launch(share=True)

