# rag_faiss.py
import json
import os
import asyncio
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Paths
DATASET_FILE = r"data\structured_rag_dataset.jsonl"
INDEX_PATH = "data/faiss_index"

# Simple in-memory cache to avoid duplicate lookups in short time window
_recent_queries = {}

def validate_json_line(line, line_number):
    """Validate and parse a single JSON line with better error handling."""
    try:
        return json.loads(line.strip())
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error on line {line_number}: {e}")
        print(f"Problematic line content: {repr(line)}")
        return None

def build_rag_index():
    """Load dataset, split text, embed, and save FAISS index."""
    print("🔄 Building FAISS index...")

    if not os.path.exists(DATASET_FILE):
        print(f"❌ Dataset file not found: {DATASET_FILE}")
        return

    valid_data = []
    try:
        with open(DATASET_FILE, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                item = validate_json_line(line, line_number)
                if item is None:
                    continue
                required_fields = ["content", "title", "category", "source"]
                if not all(field in item for field in required_fields):
                    print(f"❌ Missing required fields on line {line_number}: {item}")
                    continue
                valid_data.append(item)
    except Exception as e:
        print(f"❌ Error reading dataset file: {e}")
        return

    if not valid_data:
        print("❌ No valid data found in dataset file")
        return

    print(f"✅ Found {len(valid_data)} valid entries in dataset")

    docs = [
        Document(
            page_content=item["content"],
            metadata={
                "title": item["title"],
                "category": item["category"],
                "source": item["source"],
            },
        )
        for item in valid_data
    ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"✅ Created {len(chunks)} chunks from documents")

    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = FAISS.from_documents(chunks, embeddings)
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        db.save_local(INDEX_PATH)
        print("✅ FAISS index built and saved!")
    except Exception as e:
        print(f"❌ Error building FAISS index: {e}")

def load_rag_index():
    """Load FAISS index from disk."""
    if not os.path.exists(INDEX_PATH):
        print("⚠️ FAISS index not found, building new index...")
        build_rag_index()
        if not os.path.exists(INDEX_PATH):
            print("❌ Failed to build or load FAISS index.")
            return None

    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"❌ Error loading FAISS index: {e}")
        return None

def rag_lookup(query, top_k=5):
    """Retrieve top-k results for query — safe with missing metadata."""
    db = load_rag_index()
    if db is None:
        return "Sorry, I couldn't load the college information database. Please try again later."

    try:
        results = db.similarity_search(query, k=top_k)
        if not results:
            return "No information found in Shree Devi College database."

        context_parts = []
        for r in results:
            meta = r.metadata or {}
            title = meta.get("title", "Untitled Document")
            source = meta.get("source", "Unknown Source")
            category = meta.get("category", "General")

            context_parts.append(
                f"Source: {title} ({source}, Category: {category})\n"
                f"Content: {r.page_content}"
            )

        return "\n\n---\n\n".join(context_parts)

    except Exception as e:
        print(f"❌ Error during RAG lookup: {e}")
        return "Sorry, an error occurred while searching for information."

async def async_rag_lookup(query: str, top_k: int = 5, cache_ttl: float = 2.0):
    """Asynchronously retrieve with short-term caching to avoid duplicates."""
    now = asyncio.get_event_loop().time()

    # Clean expired cache entries
    expired = [q for q, (t, _) in _recent_queries.items() if now - t > cache_ttl]
    for q in expired:
        del _recent_queries[q]

    # Return cached result if available
    if query in _recent_queries:
        return _recent_queries[query][1]

    # Perform lookup
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, rag_lookup, query, top_k)
    _recent_queries[query] = (now, result)
    return result

def validate_dataset_file():
    """Validate the entire dataset file and report issues."""
    print(f"🔍 Validating dataset file: {DATASET_FILE}")
    if not os.path.exists(DATASET_FILE):
        print(f"❌ Dataset file not found: {DATASET_FILE}")
        return False

    valid_count = invalid_count = 0
    try:
        with open(DATASET_FILE, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                item = validate_json_line(line, line_number)
                if item is None:
                    invalid_count += 1
                else:
                    valid_count += 1
    except Exception as e:
        print(f"❌ Error reading dataset file: {e}")
        return False

    print(f"✅ Validation complete: {valid_count} valid entries, {invalid_count} invalid entries")
    return invalid_count == 0

if __name__ == "__main__":
    validate_dataset_file()
    build_rag_index()