# 🎓 SDC Joint AI  
> A lightning-fast RAG agent that answers every college-related question in plain English—courses, faculty, placements, scholarships, and more.

---

## 🚀 Demo
Live voice-enabled agent (LiveKit)  
🔗 [Talk to the bot](https://docs.livekit.io/agents/start/voice-ai)  
📦 [Source on GitHub](https://github.com/abdul-0-muheed/sdc-joint-ai)

---

## 📖 Overview
Prospective students bombard colleges with the same questions every year.  
SDC Joint AI ingests catalog data (courses, faculty, facilities, stats, rules) into a **searchable knowledge base** and replies with **concise, citation-ready answers**—no human intervention, zero lag.

**Target users**  
- Admissions & marketing teams  
- Student help-desk portals  
- Event organizers (freshers, open days, webinars)

**Key idea**  
Combine a local FAISS vector index with a tiny Python runtime to deliver **sub-second, offline, private** answers at campus scale.

---

## ✨ Features
- 🔍 Natural-language Q&A with source citations  
- 🗣️ Voice interface via LiveKit (WebRTC)  
- 📚 Auto-syncs with college JSONL dumps  
- 🐳 Fully containerized—one-command deploy  
- 🔄 Zero-downtime CI/CD with GitHub Actions  
- 🔐 100 % on-prem data—no outbound calls  
- 📈 Built-in analytics & audit logs (Supabase)

---

## 🏗️ Architecture
Three micro-services orchestrated by Docker Compose:

1. `agent` – Python runtime (RAG loop + voice handler)  
2. `vector-db` – FAISS index served over shared volume  
3. `postgres` – Supabase PostgreSQL for metadata & logs

┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   LiveKit    │────▶│  Agent Core    │────▶│  FAISS Index  │
│  (Voice)     │     │  (Python)      │     │ (Embeddings)  │
└──────────────┘     └──────────────┘     └──────────────┘
                            │                       │
                            ▼                       ▼
                    ┌──────────────┐        ┌──────────────┐
                    │  Supabase    │        │  JSONL Corpuses
                    │ (Metadata)   │        │  (Ground truth)
                    └──────────────┘        └──────────────┘
---

## 🔑 Key Components
| File | Purpose |
|------|---------|
| `src/agent.py` | Single entry-point `ask(question: str) → Answer` |
| `src/rag_faiss.py` | Brute-force vector search fallback |
| `src/rag_optimized.py` | HNSW + metadata filter pipeline |
| `src/ingest.py` | JSONL → embeddings → FAISS & PG |
| `src/voice_handler.py` | LiveKit adapter for STT/TTS |
| `taskfile.yml` | Unified task runner (`task up`, `task test`) |

---

## 🔄 Data Flow
1. **Ingest** – `ingest.py` reads JSONL → chunks → `sentence-transformers` → FAISS + Supabase  
2. **Query** – `agent.py` embeds question → top-k retrieval → LLM synthesis → citations  
3. **Voice** – LiveKit streams audio → STT → agent → TTS → user hears answer

---

## 🧪 Tech Stack
- **Language**: Python 3.11  
- **ML**: FAISS, Sentence-Transformers, HuggingFace pipeline  
- **DB**: Supabase (PostgreSQL 15)  
- **Voice**: LiveKit Agents  
- **Ops**: Docker, Docker Compose, GitHub Actions, Task  
- **Lint/Format**: Ruff, Black, MyPy

---

## 📁 Project Structure
.
├── src
│   ├── agent.py
│   ├── rag_*.py
│   ├── ingest.py
│   └── voice_handler.py
├── data
│   └── *.jsonl          # college dumps
├── scripts
│   └── seed_supabase.py
├── tests
├── .github/workflows
├── Dockerfile
├── docker-compose.yml
├── taskfile.yml
└── pyproject.toml
---

## ⚙️ Installation & Usage
Prerequisites: Docker & Task (`sh -c "$(curl -ssL https://taskfile.dev/install.sh)"`)

bash
# 1. Clone
git clone https://github.com/abdul-0-muheed/sdc-joint-ai.git
cd sdc-joint-ai

# 2. Configure env
cp .env.example .env
# Edit .env (see section below)

# 3. Run everything
task up          # builds, starts, ingests sample data
task logs        # tail containers
Local library usage (no voice):
python
from src.agent import ask
answer = ask("Which scholarships for CS students?")
print(answer.text, answer.sources)
---

## 🔌 API / Integrations
No public HTTP API—embed as a library.  
For voice, connect your LiveKit frontend to the running agent container (`ws://localhost:7880`).

---

## 🔐 Environment Variables
| Var | Description | Example |
|-----|-------------|---------|
| `SUPABASE_URL` | Postgres endpoint | `postgresql://user:pass@db:5432/sdc` |
| `SUPABASE_SERVICE_KEY` | Backend secret | `YOUR_SERVICE_KEY` |
| `FAISS_INDEX_PATH` | Mount path inside container | `/data/faiss.index` |
| `LOG_LEVEL` | Python logging | `INFO` |
| `LIVEKIT_API_KEY` | For voice | `YOUR_LK_KEY` |
| `LIVEKIT_SECRET` | For voice | `YOUR_LK_SECRET` |

---

## 🧪 Testing & Build
bash
task lint          # ruff + mypy
task test          # pytest with coverage
task build         # multi-arch Docker image
task push          # tag & push to GHCR
CI automatically runs on every PR; images land in `ghcr.io/abdul-0-muheed/sdc-joint-ai`.

---

## 📝 Notes
- Keep JSONL files under `data/`; they are hot-reloaded on container restart.  
- FAISS index is rebuilt automatically when `md5(data/*.jsonl)` changes.  
- Voice mode needs a valid LiveKit project; disable with `VOICE_ENABLED=false`.

---

## 🤝 Contributing
1. Fork & branch (`feature/foo`)  
2. Add tests & docs  
3. Run `task lint test`  
4. Open a PR—CI will do the rest.

---

## 📄 License
MIT © 2024 SDC Joint AI Contributors

---

## 📬 Contact
Open an issue or start a discussion on GitHub.