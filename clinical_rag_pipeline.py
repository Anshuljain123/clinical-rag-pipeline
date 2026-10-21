"""
Clinical RAG Pipeline
Author: Anshul Jain | AI Implementation Consultant
Built for: Pinnacle Pain Management Specialists, Hinsdale, IL

Pipeline flow:
  Patient notes (.txt) -> chunk & embed -> ChromaDB vector store
  -> similarity retrieval -> structured LLM output (Claude API)

Output format: Diagnosis / Medications / Vitals / Recommended Action
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict

from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from sentence_transformers import SentenceTransformer
import anthropic

# ─── Configuration ────────────────────────────────────────────────────────────

CHROMA_PERSIST_PATH = "./chroma_store"
COLLECTION_NAME     = "patient_notes"
EMBED_MODEL         = "all-MiniLM-L6-v2"   # lightweight, runs locally
CHUNK_SIZE          = 500
CHUNK_OVERLAP       = 50
TOP_K               = 5

SYSTEM_PROMPT = """You are a clinical documentation assistant working with pain management patient records.

Given retrieved excerpts from a patient's clinical notes, produce a concise structured summary 
in the following JSON format only — no preamble, no explanation:

{
  "Diagnosis": "Primary diagnosis or chief complaint",
  "Medications": ["Med 1 - dose/frequency", "Med 2 - dose/frequency"],
  "Vitals": {
    "BP": "value or Not documented",
    "HR": "value or Not documented",
    "SpO2": "value or Not documented"
  },
  "Recommended_Action": "Next steps or treatment plan"
}

Rules:
- Base your response ONLY on the provided context excerpts.
- If a field is not mentioned, use "Not documented".
- Do not infer, invent, or extrapolate beyond what is in the notes.
- Medications list may be empty [] if none are mentioned.
"""

# ─── Embedding ────────────────────────────────────────────────────────────────

_embedder = None

def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        print(f"[Embed] Loading model: {EMBED_MODEL}")
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder

def embed(texts: List[str]) -> List[List[float]]:
    return get_embedder().encode(texts, convert_to_numpy=True).tolist()

# ─── ChromaDB ─────────────────────────────────────────────────────────────────

def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

# ─── Ingestion ────────────────────────────────────────────────────────────────

def ingest_notes(notes_dir: str) -> None:
    """
    Load all .txt patient note files from a directory.
    Each filename is treated as the patient ID (e.g., PT_00123.txt).
    Chunks the text, embeds it, and stores it in ChromaDB.
    """
    collection = get_collection()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    note_files = list(Path(notes_dir).glob("*.txt"))
    if not note_files:
        print(f"[Ingest] No .txt files found in: {notes_dir}")
        return

    for note_path in note_files:
        patient_id = note_path.stem
        raw_text   = note_path.read_text(encoding="utf-8")
        chunks     = splitter.split_text(raw_text)

        if not chunks:
            print(f"[Ingest] Skipping empty file: {note_path.name}")
            continue

        embeddings = embed(chunks)
        ids        = [f"{patient_id}__chunk_{i}" for i in range(len(chunks))]
        metadatas  = [{"patient_id": patient_id, "chunk_index": i} for i in range(len(chunks))]

        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )
        print(f"[Ingest] {patient_id}: {len(chunks)} chunks stored")

    print(f"\n[Ingest] Done. {len(note_files)} patient files ingested.")

# ─── Retrieval ────────────────────────────────────────────────────────────────

def retrieve(query: str, patient_id: str = None) -> List[str]:
    """
    Embed the query string and retrieve the top-K most similar chunks
    from ChromaDB. Optionally filter by patient_id.
    """
    collection    = get_collection()
    query_vector  = embed([query])[0]
    where_filter  = {"patient_id": patient_id} if patient_id else None

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=TOP_K,
        where=where_filter
    )

    chunks = results.get("documents", [[]])[0]
    if not chunks:
        print(f"[Retrieve] No results found for patient_id='{patient_id}'")
    return chunks

# ─── LLM Summary ──────────────────────────────────────────────────────────────

def generate_summary(chunks: List[str], patient_id: str = "Unknown") -> Dict:
    """
    Pass retrieved context chunks to Claude API.
    Returns a structured clinical summary dict.
    """
    if not chunks:
        return {
            "Diagnosis": "No data found",
            "Medications": [],
            "Vitals": {"BP": "Not documented", "HR": "Not documented", "SpO2": "Not documented"},
            "Recommended_Action": "No data found"
        }

    context      = "\n\n---\n\n".join(chunks)
    user_message = f"Patient ID: {patient_id}\n\nClinical Note Excerpts:\n{context}"

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set. Export it before running.")

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}]
    )

    raw = response.content[0].text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Return raw if JSON parse fails (shouldn't happen with this prompt)
        return {"raw_output": raw}

# ─── End-to-end pipeline ──────────────────────────────────────────────────────

def run_pipeline(patient_id: str, query: str = "Summarize this patient's clinical status") -> Dict:
    """
    Full pipeline for a single patient:
      1. Retrieve top-K relevant chunks from vector store
      2. Send to Claude for structured clinical summary
    """
    print(f"\n[Pipeline] Patient: {patient_id}")
    print(f"[Pipeline] Query: {query}")

    chunks  = retrieve(query, patient_id=patient_id)
    print(f"[Pipeline] Retrieved {len(chunks)} chunks")

    summary = generate_summary(chunks, patient_id=patient_id)
    return summary

# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Clinical RAG Pipeline — Pinnacle Pain Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest all patient notes from a folder
  python clinical_rag_pipeline.py ingest --dir ./patient_notes

  # Query summary for a specific patient
  python clinical_rag_pipeline.py query --patient PT_00123

  # Custom query
  python clinical_rag_pipeline.py query --patient PT_00123 --query "What medications is this patient on?"
        """
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest patient note files into vector store")
    p_ingest.add_argument("--dir", required=True, help="Folder containing .txt patient note files")

    # query
    p_query = sub.add_parser("query", help="Generate structured summary for a patient")
    p_query.add_argument("--patient", required=True, help="Patient ID (the filename without .txt)")
    p_query.add_argument(
        "--query",
        default="Summarize this patient's clinical status including diagnosis, medications, vitals, and recommended next steps.",
        help="Query string for vector similarity search"
    )

    args = parser.parse_args()

    if args.command == "ingest":
        ingest_notes(args.dir)

    elif args.command == "query":
        result = run_pipeline(args.patient, args.query)
        print("\n── Clinical Summary " + "─" * 50)
        print(json.dumps(result, indent=2))
        print("─" * 70)


if __name__ == "__main__":
    main()
