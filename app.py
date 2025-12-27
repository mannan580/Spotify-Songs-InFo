import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
import gradio as gr

# ----------------------------
# Load Groq API Key in Colab
# ----------------------------
from google.colab import userdata
GROQ_API_KEY = userdata.get('GROQ_API_KEY')
client = Groq(api_key=GROQ_API_KEY)

# ----------------------------
# Load CSV from Google Drive
# ----------------------------
CSV_URL = "https://drive.google.com/uc?id=13vzqtHJs9WHFpiCjDRgUjoctP7P3FIn6&export=download"
df = pd.read_csv(CSV_URL)
df['combined_text'] = df.astype(str).agg(' '.join, axis=1)
texts = df['combined_text'].tolist()

# ----------------------------
# Chunk text
# ----------------------------
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

all_chunks = []
for t in texts:
    all_chunks.extend(chunk_text(t))

# ----------------------------
# Embeddings using sentence-transformers
# ----------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(all_chunks, convert_to_numpy=True, show_progress_bar=True)
embedding_dim = embeddings.shape[1]

# ----------------------------
# FAISS index
# ----------------------------
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings)

# ----------------------------
# RAG query function
# ----------------------------
def rag_query(user_query, top_k=3):
    query_emb = model.encode([user_query], convert_to_numpy=True)
    D, I = index.search(query_emb, top_k)
    retrieved_chunks = [all_chunks[i] for i in I[0]]
    context = "\n".join(retrieved_chunks)

    prompt = f"Context: {context}\n\nQuestion: {user_query}\nAnswer:"
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile"
    )
    return response.choices[0].message.content

# ----------------------------
# Advanced Gradio UI
# ----------------------------
with gr.Blocks(
    theme=gr.themes.Monochrome(
        primary_hue="neutral",
        secondary_hue="neutral",
        neutral_hue="slate"
    ),
    css="""
    body {
        background-color: #0f0f0f;
    }
    .gradio-container {
        max-width: 1100px;
        margin: auto;
    }
    """
) as demo:

    # Header
    gr.Markdown(
        """
        <h1 style="text-align:center; color:white; font-weight:600;">
            Document Question Answering System
        </h1>
        <p style="text-align:center; color:#b0b0b0;">
            Retrieval-Augmented Generation powered by FAISS and Groq
        </p>
        <hr style="border:1px solid #333;">
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            user_input = gr.Textbox(
                label="Query",
                placeholder="Enter your question about the document...",
                lines=4
            )

            submit_btn = gr.Button(
                "Submit",
                variant="primary"
            )

        with gr.Column(scale=2):
            output = gr.Textbox(
                label="Answer",
                placeholder="The response will appear here...",
                lines=20,
                max_lines=80,
                interactive=False
            )

    gr.Markdown(
        """
        <hr style="border:1px solid #333;">
        <p style="text-align:center; color:#888; font-size:12px;">
            Built with FAISS Vector Search · SentenceTransformers · Groq LLM
        </p>
        """
    )

    submit_btn.click(fn=rag_query, inputs=user_input, outputs=output)

demo.launch()
