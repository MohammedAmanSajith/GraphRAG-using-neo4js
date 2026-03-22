import os
import argparse
import re
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from pathlib import Path
from tqdm import tqdm
import numpy as np
from FlagEmbedding import BGEM3FlagModel

load_dotenv()

# ── Neo4j ─────────────────────────────────────────────────────────────────────
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database=os.getenv("NEO4J_DATABASE")
)

# ── BGE-M3 Embeddings (local, no API quota) ───────────────────────────────────
print("Loading BGE-M3 embedding model...")
bge_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
print("✓ BGE-M3 loaded")

def embed_texts(texts: list) -> list:
    """Embed a list of texts using BGE-M3, returns list of vectors."""
    result = bge_model.encode(texts, batch_size=12, max_length=512)
    return result["dense_vecs"].tolist()

def embed_query(text: str) -> list:
    """Embed a single query using BGE-M3."""
    result = bge_model.encode([text], batch_size=1, max_length=512)
    return result["dense_vecs"][0].tolist()

# ── LLM (Groq first, OpenRouter second, Gemini fallback) ─────────────────────
# Each entry: (provider, model_name)
AVAILABLE_MODELS = [
    ("groq",         "llama-3.3-70b-versatile"),
    ("groq",         "llama3-70b-8192"),
    ("groq",         "llama3-8b-8192"),
    ("groq",         "gemma2-9b-it"),
    ("openrouter",   "meta-llama/llama-3.3-70b-instruct:free"),
    ("openrouter",   "mistralai/mistral-small-3.1-24b-instruct:free"),
    ("openrouter",   "nousresearch/hermes-3-llama-3.1-405b:free"),
    ("gemini",       "gemini-2.0-flash-lite"),
    ("gemini",       "gemini-2.0-flash"),
    ("gemini",       "gemini-2.5-flash"),
]

current_model_index = 0

def get_llm(index=None):
    if index is None:
        index = current_model_index
    provider, model_name = AVAILABLE_MODELS[index]
    if provider == "groq":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            openai_api_key=os.getenv("GROQ_API_KEY"),
            openai_api_base="https://api.groq.com/openai/v1",
            temperature=0
        )
    elif provider == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0,
            default_headers={
                "HTTP-Referer": "https://github.com/graphrag",
                "X-Title": "GraphRAG"
            }
        )
    else:
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0
        )

def rotate_model():
    global current_model_index, llm, transformer
    current_model_index = (current_model_index + 1) % len(AVAILABLE_MODELS)
    provider, model_name = AVAILABLE_MODELS[current_model_index]
    print(f"\n  \u21bb Switching to [{provider}] {model_name}")
    llm = get_llm(current_model_index)
    transformer = _build_transformer(llm)
    return model_name

def get_retry_delay(err: str, default: float = 15.0) -> float:
    for pattern in [r"retryDelay.*?(\d+)s", r"retry in (\d+\.?\d*)s"]:
        m = re.search(pattern, err)
        if m:
            return float(m.group(1))
    return default

def _build_transformer(llm_instance):
    provider = AVAILABLE_MODELS[current_model_index][0]
    use_prompt = provider == "groq"
    return LLMGraphTransformer(
        llm=llm_instance,
        allowed_nodes=[
            "Person", "Organization", "Location", "Concept",
            "Event", "Product", "Technology", "Field", "Award", "Element"
        ],
        allowed_relationships=[
            "WORKS_AT", "FOUNDED", "LOCATED_IN", "PART_OF", "RELATED_TO",
            "DISCOVERED", "INVENTED", "WON", "PARTICIPATED_IN", "LEADS",
            "COLLABORATED_WITH", "STUDIED_AT", "PUBLISHED", "DEVELOPED",
            "INFLUENCED", "CAUSED", "USES", "BELONGS_TO", "KNOWN_FOR", "AFFILIATED_WITH"
        ],
        node_properties=False if use_prompt else ["description"],
        relationship_properties=False,
        strict_mode=False,
        ignore_tool_usage=use_prompt
    )

llm = get_llm(0)
transformer = _build_transformer(llm)
print(f"✓ LLM ready: [{AVAILABLE_MODELS[0][0]}] {AVAILABLE_MODELS[0][1]}")

PDF_FOLDER = "pdfs"

# ── PDF Loading & Chunking ────────────────────────────────────────────────────
def create_pdf_folder():
    Path(PDF_FOLDER).mkdir(exist_ok=True)
    print(f"✓ PDF folder ready at: {PDF_FOLDER}/")

def load_and_chunk_pdfs():
    pdf_files = list(Path(PDF_FOLDER).glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {PDF_FOLDER}/")
        return []
    print(f"Found {len(pdf_files)} PDF file(s)\n")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = []
    for pdf in tqdm(pdf_files, desc="Loading PDFs", unit="file"):
        pages = PyPDFLoader(str(pdf)).load()
        docs.extend(splitter.split_documents(pages))
    print(f"\nTotal chunks: {len(docs)}\n")
    return docs

# ── Step 1: LLM Extraction → Knowledge Graph ─────────────────────────────────
def scan_and_convert_pdfs():
    print("\n" + "="*50)
    print("STEP 1: LLM EXTRACTION → KNOWLEDGE GRAPH")
    print("Directed | Unweighted | Cyclic | Multi-edge")
    print("="*50 + "\n")

    documents = load_and_chunk_pdfs()
    if not documents:
        return

    print("Extracting entities & relationships via LLM...\n")
    batch_size = 1
    total_batches = len(documents)
    max_retries = len(AVAILABLE_MODELS)

    with tqdm(total=total_batches, desc="Processing batches", unit="batch") as pbar:
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            success = False

            for attempt in range(max_retries):
                try:
                    graph_docs = transformer.convert_to_graph_documents(batch)
                    graph.add_graph_documents(graph_docs, include_source=True)
                    pbar.set_postfix({"model": AVAILABLE_MODELS[current_model_index]})
                    success = True
                    time.sleep(8)
                    break
                except Exception as e:
                    err = str(e)
                    if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower() or "rate-limited" in err.lower() or "rate_limit" in err.lower():
                        delay = get_retry_delay(err)
                        print(f"\n  \u26a0 Rate limited on batch {batch_num}, waiting {delay:.0f}s then retrying same model...")
                        time.sleep(delay)
                    elif "404" in err or "NOT_FOUND" in err or "not found" in err.lower() or "not enabled" in err.lower() or "model_not_found" in err.lower() or "does not exist" in err.lower() or "402" in err or "json_schema" in err or "response_format" in err:
                        print(f"\n  \u26a0 Model unavailable, rotating...")
                        rotate_model()
                    else:
                        print(f"\n  \u2717 Batch {batch_num} error: {e}")
                        break

            if not success:
                print(f"  \u2717 Skipping batch {batch_num} after {max_retries} attempts")
            pbar.update(1)

    print("\nGenerating BGE-M3 embeddings for semantic search...")
    _store_node_embeddings()
    print("\n✓ Knowledge graph built successfully!")
    print_schema()

# ── Step 2: Store BGE-M3 Embeddings on Nodes ─────────────────────────────────
def _store_node_embeddings():
    nodes = graph.query(
        "MATCH (n) WHERE n.id IS NOT NULL "
        "RETURN n.id AS id, labels(n)[0] AS label, n.description AS desc"
    )
    if not nodes:
        print("  No nodes found to embed.")
        return

    texts = [f"{n['label']}: {n['id']}. {n['desc'] or ''}" for n in nodes]
    try:
        vecs = embed_texts(texts)
        for node, vec in zip(nodes, vecs):
            graph.query(
                "MATCH (n {id: $id}) SET n.embedding = $embedding",
                {"id": node["id"], "embedding": vec}
            )
        print(f"  ✓ BGE-M3 embeddings stored for {len(nodes)} nodes")
    except Exception as e:
        print(f"  \u2717 Embedding error: {e}")

# ── Step 3: Semantic Search (BGE-M3 cosine similarity) ───────────────────────
def semantic_search(question: str, top_k: int = 5) -> list:
    try:
        q_vec = np.array(embed_query(question))
        nodes = graph.query(
            "MATCH (n) WHERE n.embedding IS NOT NULL AND n.id IS NOT NULL "
            "RETURN n.id AS id, labels(n)[0] AS label, n.description AS desc, n.embedding AS embedding"
        )
        if not nodes:
            return []

        scored = []
        for node in nodes:
            n_vec = np.array(node["embedding"])
            sim = float(np.dot(q_vec, n_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(n_vec) + 1e-9))
            scored.append((sim, node))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in scored[:top_k]]

    except Exception as e:
        print(f"  \u2717 Semantic search error: {e}")
        return graph.query(
            "MATCH (n) WHERE n.id IS NOT NULL "
            "RETURN n.id AS id, labels(n)[0] AS label, n.description AS desc LIMIT $k",
            {"k": top_k}
        )

# ── Step 4: Graph Traversal ───────────────────────────────────────────────────
def graph_traversal(entity_ids: list, depth: int = 2) -> list:
    if not entity_ids:
        return []
    return graph.query(
        """
        UNWIND $ids AS seed
        MATCH path = (start {id: seed})-[*1..$depth]->(end)
        UNWIND relationships(path) AS r
        RETURN DISTINCT
            startNode(r).id          AS from_node,
            labels(startNode(r))[0]  AS from_type,
            type(r)                  AS relationship,
            endNode(r).id            AS to_node,
            labels(endNode(r))[0]    AS to_type,
            endNode(r).description   AS to_desc
        """,
        {"ids": entity_ids, "depth": depth}
    )

# ── Full RAG Pipeline ─────────────────────────────────────────────────────────
def rag_pipeline(question: str) -> str:
    # 1. BGE-M3 semantic search → seed entities
    print("  [1/3] BGE-M3 semantic search...")
    seed_nodes = semantic_search(question, top_k=5)
    seed_ids = [n["id"] for n in seed_nodes]
    print(f"        Seeds: {seed_ids}")

    # 2. Graph traversal from seeds
    print("  [2/3] Graph traversal (directed, multi-edge, cyclic)...")
    traversal = graph_traversal(seed_ids, depth=2)

    context_parts = []
    for r in traversal:
        line = (f"({r['from_type']}:{r['from_node']}) -[{r['relationship']}]-> "
                f"({r['to_type']}:{r['to_node']})")
        if r.get("to_desc"):
            line += f" | {r['to_desc']}"
        context_parts.append(line)

    for n in seed_nodes:
        if n.get("desc"):
            context_parts.append(f"{n['label']}:{n['id']} — {n['desc']}")

    context = "\n".join(context_parts) if context_parts else "No graph context found."

    # 3. LLM answer synthesis
    print("  [3/3] LLM answer synthesis...")
    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template="""You are a knowledge graph assistant. Use the graph context to answer accurately.

Graph Context:
{context}

Question: {question}

Answer concisely based on the graph context. If insufficient, say so."""
    )

    chain = prompt | llm
    for attempt in range(len(AVAILABLE_MODELS)):
        try:
            response = chain.invoke({"question": question, "context": context})
            break
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "404" in str(e) or "NOT_FOUND" in str(e):
                rotate_model()
                chain = prompt | llm
            else:
                raise

    if hasattr(response, "content"):
        content = response.content
        if isinstance(content, list) and content:
            item = content[0]
            return item["text"] if isinstance(item, dict) and "text" in item else str(item)
        return str(content)
    return str(response)

# ── Schema ────────────────────────────────────────────────────────────────────
def print_schema():
    print("\n" + "="*50)
    print("KNOWLEDGE GRAPH SCHEMA")
    print("="*50)
    try:
        nodes = graph.query("MATCH (n) RETURN DISTINCT labels(n) AS NodeType, count(*) AS Count")
        print("\nNode Types:")
        for n in nodes:
            print(f"  - {n['NodeType']}: {n['Count']} nodes")

        rels = graph.query("MATCH ()-[r]->() RETURN DISTINCT type(r) AS RelType, count(*) AS Count")
        print("\nRelationship Types (Directed):")
        for r in rels:
            print(f"  - {r['RelType']}: {r['Count']} edges")

        samples = graph.query(
            "MATCH (a)-[r]->(b) RETURN labels(a)[0] AS From, type(r) AS Rel, labels(b)[0] AS To LIMIT 8"
        )
        print("\nSample Directed Edges:")
        for s in samples:
            print(f"  ({s['From']})-[{s['Rel']}]->({s['To']})")
    except Exception as e:
        print(f"Error: {e}")

def get_schema_summary():
    try:
        nodes = graph.query("MATCH (n) RETURN DISTINCT labels(n) AS NodeType")
        node_types = [n["NodeType"][0] for n in nodes if n["NodeType"]]
        rels = graph.query("MATCH ()-[r]->() RETURN DISTINCT type(r) AS RelType")
        rel_types = [r["RelType"] for r in rels]
        return f"Nodes: {', '.join(node_types)}\nRelationships: {', '.join(rel_types)}"
    except:
        return "Schema unavailable. Please scan PDFs first."

# ── Chatbot ───────────────────────────────────────────────────────────────────
def chatbot():
    print("\n" + "="*50)
    print("GRAPH RAG CHATBOT")
    print("BGE-M3 Semantic Search → Graph Traversal → LLM Answer")
    print("="*50)
    print("Type 'exit' or 'quit' to return to main menu\n")
    print(f"Graph loaded:\n{get_schema_summary()}\n")
    print("="*50 + "\n")

    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ["exit", "quit", "back"]:
            break
        if not question:
            continue
        print()
        try:
            answer = rag_pipeline(question)
            print(f"\nAssistant: {answer}")
        except Exception as e:
            print(f"\n\u2717 Error: {e}")

# ── Delete Graph Data ─────────────────────────────────────────────────────────
def delete_graph_data():
    print("\n" + "="*50)
    print("DELETE ALL GRAPH DATA")
    print("="*50)
    confirm = input("\n\u26a0 This will delete ALL nodes, relationships and embeddings.\nType 'yes' to confirm: ").strip().lower()
    if confirm != "yes":
        print("Cancelled.")
        return
    try:
        graph.query("MATCH (n) DETACH DELETE n")
        print("✓ All graph data deleted. You can now re-scan your PDFs.")
    except Exception as e:
        print(f"\u2717 Error: {e}")

# ── Main Menu ─────────────────────────────────────────────────────────────────
def main_menu():
    create_pdf_folder()
    while True:
        print("\n" + "="*50)
        print("GRAPH RAG SYSTEM")
        print("LLM: Groq → OpenRouter → Gemini | Storage: Neo4j | Embeddings: BGE-M3")
        print("="*50)
        print("\n1. Scan PDFs → Build Knowledge Graph")
        print("2. Chatbot   → Query the Graph")
        print("3. Show Schema")
        print("4. Delete All Graph Data")
        print("5. Exit")

        choice = input("\nEnter your choice (1-5): ").strip()
        if choice == "1":
            scan_and_convert_pdfs()
        elif choice == "2":
            chatbot()
        elif choice == "3":
            print_schema()
        elif choice == "4":
            delete_graph_data()
        elif choice == "5":
            print("\nGoodbye!")
            break
        else:
            print("\n\u2717 Invalid choice.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph RAG System")
    parser.add_argument("--console", action="store_true", help="Run in console mode")
    args = parser.parse_args()
    if args.console:
        main_menu()
    else:
        print("Use --console flag to run the application")
        print("Example: python graph_rag.py --console")
