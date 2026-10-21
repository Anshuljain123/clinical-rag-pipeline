# Clinical RAG Pipeline

A document-ingestion-to-summary pipeline built for a regulated pain management clinic.

**Stack:** Python · LangChain · ChromaDB · SentenceTransformer · Claude API (Anthropic)

---

## What It Does

Loads unstructured patient clinical notes, chunks and embeds them into a local vector store, retrieves the most relevant context via semantic similarity search, and produces a structured clinical summary through an LLM.

```
Patient notes (.txt)
    → chunk & embed (SentenceTransformer: all-MiniLM-L6-v2)
    → ChromaDB vector store (cosine similarity, persistent)
    → similarity retrieval (top-K chunks)
    → Claude API (structured output)
    → { Diagnosis / Medications / Vitals / Recommended Action }
```

---

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
```

---

## Usage

**Step 1 — Ingest patient notes**

Place `.txt` patient note files in a folder (filename = patient ID):

```
patient_notes/
  PT_00123.txt
  PT_00456.txt
```

```bash
python clinical_rag_pipeline.py ingest --dir ./patient_notes
```

**Step 2 — Query a patient summary**

```bash
python clinical_rag_pipeline.py query --patient PT_00123
```

**Output:**

```json
{
  "Diagnosis": "Lumbar radiculopathy, L4-L5",
  "Medications": [
    "Gabapentin 300mg - three times daily",
    "Meloxicam 15mg - once daily with food"
  ],
  "Vitals": {
    "BP": "128/82",
    "HR": "74",
    "SpO2": "98%"
  },
  "Recommended_Action": "Schedule follow-up lumbar epidural steroid injection. Monitor for GI side effects from Meloxicam."
}
```

**Custom query:**

```bash
python clinical_rag_pipeline.py query --patient PT_00123 --query "What procedures has this patient had?"
```

---

## Design Notes

- **Local embeddings** — SentenceTransformer runs locally; no PHI leaves the environment during embedding
- **Persistent vector store** — ChromaDB stores embeddings to disk; no re-ingestion needed between sessions
- **Patient-scoped retrieval** — queries are filtered by `patient_id` metadata, preventing cross-patient bleed
- **HIPAA considerations** — designed for on-premise or private cloud deployment; LLM API call is the only external touch point
- **Swappable LLM** — Claude API can be replaced with Azure OpenAI or a local model by changing `generate_summary()`

---

## Built For

  
AI Implementation Consulting · Anshul Jain
