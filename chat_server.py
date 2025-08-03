# chat_server.py
import os
import json
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHUNK_SIZE = 500
VECTOR_FILE = "vectors.json"
KNOWLEDGE_FILE = "knowledge.txt"

# Globals to hold data
vector_store = []
knowledge_chunks = []

# --- Text Chunking ---
def chunk_text(text, size=CHUNK_SIZE):
    import re
    sentences = re.split(r'(?<=[.?!])\s+', text)
    chunks = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) > size:
            chunks.append(current.strip())
            current = sentence
        else:
            current += " " + sentence
    if current.strip():
        chunks.append(current.strip())
    return chunks

# --- Cosine Similarity ---
def cosine_similarity(vecA, vecB):
    a = np.array(vecA)
    b = np.array(vecB)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# --- Load or Generate Embeddings ---
def load_or_generate_vectors():
    global knowledge_chunks, vector_store

    if os.path.exists(VECTOR_FILE):
        with open(VECTOR_FILE, "r", encoding="utf-8") as f:
            vector_store = json.load(f)
            knowledge_chunks = [v["text"] for v in vector_store]
        print("✅ Loaded cached vectors.")
        return

    if not os.path.exists(KNOWLEDGE_FILE):
        raise FileNotFoundError("❌ knowledge.txt not found.")

    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    knowledge_chunks = chunk_text(text)
    vector_store = []

    for chunk in knowledge_chunks:
        embedding = openai.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )
        vector_store.append({
            "text": chunk,
            "embedding": embedding.data[0].embedding
        })
        print(f"🔹 Embedded chunk: {chunk[:60]}...")

    with open(VECTOR_FILE, "w", encoding="utf-8") as f:
        json.dump(vector_store, f, indent=2)
    print("✅ Saved embeddings to vectors.json")

# --- Get Answer from OpenAI with Context ---
def get_answer(question):
    # Step 1: Embed the question
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )
    question_embedding = response.data[0].embedding

    # Step 2: Find most relevant chunk
    scored_chunks = sorted(
        [
            {
                "text": entry["text"],
                "score": cosine_similarity(question_embedding, entry["embedding"])
            }
            for entry in vector_store
        ],
        key=lambda x: x["score"],
        reverse=True
    )

    top_chunk = scored_chunks[0]["text"]

    # Step 3: Use Chat API
    prompt = f"Knowledge:\n{top_chunk}\n\nQuestion:\n{question}"

    completion = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You're an assistant that answers questions based on the provided knowledge and general information. Use the knowledge first if it's relevant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return completion.choices[0].message.content.strip()
