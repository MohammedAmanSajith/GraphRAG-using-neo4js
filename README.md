# Graph RAG System with Neo4j, Groq and BGE-M3

A Graph-based Retrieval Augmented Generation (RAG) system that converts PDF documents into knowledge graphs using Neo4j, Groq LLM, and local BGE-M3 embeddings.

## How It Works

```
PDF → Chunks → Groq LLM → Entities & Relationships → Neo4j Graph
Question → BGE-M3 Embedding → Semantic Search → Graph Traversal → Groq LLM → Answer
```

## Prerequisites

- Python 3.8+
- Neo4j Aura account — https://console.neo4j.io (free tier)
- Groq API key — https://console.groq.com (free tier)
- OpenRouter API key — https://openrouter.ai (free tier, fallback)
- Google Gemini API key — https://ai.google.dev (free tier, final fallback)

## Project Structure

```
Graphrag/
├── graph_rag.py       # Main application
├── requirements.txt   # Python dependencies
├── .env               # API keys and credentials (DO NOT COMMIT)
├── .gitignore
├── pdfs/              # Place your PDF files here
└── README.md
```

## Setup

**Step 1 — Create and activate virtual environment**
```
python -m venv graph_venv
graph_venv\Scripts\activate
```

**Step 2 — Install dependencies**
```
pip install -r requirements.txt
pip install transformers==4.44.2
```

**Step 3 — Create your `.env` file**

Create a file named `.env` in the project root:
```
NEO4J_URI=your_neo4j_uri
NEO4J_USERNAME=your_username
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=your_database
GROQ_API_KEY=your_groq_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
GOOGLE_API_KEY=your_google_api_key
```

Where to get credentials:
- Neo4j: https://console.neo4j.io → Create free instance → copy URI, username, password, database
- Groq: https://console.groq.com → API Keys → Create key
- OpenRouter: https://openrouter.ai → Keys → Create key
- Google Gemini: https://ai.google.dev → Get API key

**Step 4 — Add PDFs**

Place your PDF files inside the `pdfs/` folder.

**Step 5 — Run the application**
```
python graph_rag.py --console
```

## Usage

### Build the Knowledge Graph

```
1. Scan PDFs → Build Knowledge Graph
```
- Loads all PDFs from the `pdfs/` folder
- Splits into chunks of 1000 characters
- Extracts entities and relationships via Groq LLM
- Stores graph in Neo4j
- Generates BGE-M3 embeddings on all nodes

### Query the Graph

```
2. Chatbot → Query the Graph
```
- Type any question in natural language
- BGE-M3 finds the most relevant nodes (semantic search)
- Graph traversal expands context (depth 2)
- Groq LLM synthesizes the final answer

### Other Options

```
3. Show Schema        — view node types, relationship types, sample edges
4. Delete All Data    — wipe Neo4j graph (use before re-scanning)
5. Exit
```

## Adding New PDFs or Updating Existing Ones

1. Activate the virtual environment
2. Place new PDF files into the `pdfs/` folder
3. Run `python graph_rag.py --console`
4. Select option `4` to delete old graph data (if re-scanning everything) → type `yes`
5. Select option `1` to scan and rebuild the graph

## Updating the Code

1. Activate the virtual environment
2. Pull latest changes — `git pull`
3. Install any new dependencies — `pip install -r requirements.txt`
4. Run `python graph_rag.py --console`

## LLM Provider Chain

The system tries providers in this order, rotating automatically on errors:

| Priority | Provider   | Models                                                      |
|----------|------------|-------------------------------------------------------------|
| 1st      | Groq       | llama-3.3-70b-versatile, llama3-70b, llama3-8b, gemma2-9b  |
| 2nd      | OpenRouter | llama-3.3-70b, mistral-small-3.1-24b, hermes-3-405b        |
| 3rd      | Gemini     | gemini-2.0-flash-lite, gemini-2.0-flash, gemini-2.5-flash  |

## Graph Schema

Node types extracted: `Person`, `Organization`, `Location`, `Concept`, `Event`, `Product`, `Technology`, `Field`, `Award`, `Element`

Relationship types: `WORKS_AT`, `FOUNDED`, `LOCATED_IN`, `PART_OF`, `RELATED_TO`, `DISCOVERED`, `INVENTED`, `WON`, `PARTICIPATED_IN`, `LEADS`, `COLLABORATED_WITH`, `STUDIED_AT`, `PUBLISHED`, `DEVELOPED`, `INFLUENCED`, `CAUSED`, `USES`, `BELONGS_TO`, `KNOWN_FOR`, `AFFILIATED_WITH`

## Troubleshooting

**BGE-M3 fails to load**
```
pip install transformers==4.44.2
```

**Neo4j connection error**
- Check your instance is running at https://console.neo4j.io
- Verify credentials in `.env`
- Wait 60 seconds after creating a new instance

**All batches skipped / no nodes created**
- Run option `4` to delete empty graph data, then re-scan
- Check API keys in `.env` are valid
- Groq free tier: 14,400 requests/day on llama-3.3-70b

**Rate limits**
- The system waits and retries automatically on 429 errors
- It rotates to the next provider on quota/credit errors

## Security

Never commit your `.env` file. The `.gitignore` excludes:
- `.env`
- `graph_venv/`
- `__pycache__/`
- `*.pyc`
- `.ipynb_checkpoints/`

## Dependencies

- [LangChain](https://langchain.com/) — LLM orchestration
- [Neo4j](https://neo4j.com/) — Graph database
- [Groq](https://groq.com/) — Fast LLM inference
- [BGE-M3](https://huggingface.co/BAAI/bge-m3) — Local embeddings (no API quota)
- [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) — BGE-M3 wrapper
