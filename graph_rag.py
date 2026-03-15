import os
import argparse
import re
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from pathlib import Path
from tqdm import tqdm
import numpy as np

load_dotenv()

# ── Neo4j connection ──────────────────────────────────────────────────────────
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database=os.getenv("NEO4J_DATABASE")
)

# ── LLM + Embeddings ──────────────────────────────────────────────────────────
# Models ordered by preference; rotation happens on 429 RESOURCE_EXHAUSTED
AVAILABLE_MODELS = [
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-pro-latest",
]

current_model_index = 0

def get_llm(model_name=None):
    if model_name is None:
        model_name = AVAILABLE_MODELS[current_model_index]
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0
    )

def rotate_model():
    """Switch to the next available model on quota exhaustion."""
    global current_model_index, llm, transformer
    current_model_index = (current_model_index + 1) % len(AVAILABLE_MODELS)
    new_model = AVAILABLE_MODELS[current_model_index]
    print(f"\n  \u21bb Switching to model: {new_model}")
    llm = get_llm(new_model)
    transformer = _build_transformer(llm)
    return new_model

def get_retry_delay(err: str, default: float = 10.0) -> float:
    """Extract retryDelay seconds from error message, fallback to default."""
    match = re.search(r'retryDelay.*?(\d+)s', err)
    if match:
        return float(match.group(1))
    match = re.search(r'retry in (\d+\.?\d*)s', err)
    if match:
        return float(match.group(1))
    return default

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

def _build_transformer(llm_instance):
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
        node_properties=["description"],
        relationship_properties=False,
        strict_mode=False
    )

llm = get_llm()
embeddings = get_embeddings()

# ── LLMGraphTransformer: Directed, Unweighted, Cyclic, Multi-edge graph ───────
transformer = _build_transformer(llm)

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
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = []
    for pdf in tqdm(pdf_files, desc="Loading PDFs", unit="file"):
        pages = PyPDFLoader(str(pdf)).load()
        docs.extend(splitter.split_documents(pages))
    print(f"\nTotal chunks: {len(docs)}\n")
    return docs

# ── Step 1: LLM-Based Extraction → Knowledge Graph ───────────────────────────
def scan_and_convert_pdfs():
    print("\n" + "="*50)
    print("STEP 1: LLM EXTRACTION → KNOWLEDGE GRAPH")
    print("Directed | Unweighted | Cyclic | Multi-edge")
    print("="*50 + "\n")

    documents = load_and_chunk_pdfs()
    if not documents:
        return

    # Store chunk text on nodes for semantic search later
    print("Extracting entities & relationships via LLM...\n")
    batch_size = 5
    total_batches = (len(documents) + batch_size - 1) // batch_size
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
                    time.sleep(3)
                    break
                except Exception as e:
                    err = str(e)
                    if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
                        delay = get_retry_delay(err, default=15.0)
                        print(f"\n  ⚠ Quota on batch {batch_num}, switching model, waiting {delay:.0f}s...")
                        rotate_model()
                        time.sleep(delay)
                    elif "404" in err or "NOT_FOUND" in err:
                        print(f"\n  ⚠ Model not found, rotating...")
                        rotate_model()
                    else:
                        print(f"\n  ✗ Batch {batch_num} error: {e}")
                        break

            if not success:
                print(f"  ✗ Skipping batch {batch_num} after {max_retries} attempts")
            pbar.update(1)

    # Store embeddings on each node for semantic search
    print("\nGenerating embeddings for semantic search...")
    _store_node_embeddings()

    print("\n✓ Knowledge graph built successfully!")
    print_schema()

def _store_node_embeddings():
    """Embed each node's id+description and store as property in Neo4j."""
    nodes = graph.query("MATCH (n) WHERE n.id IS NOT NULL RETURN n.id AS id, labels(n)[0] AS label, n.description AS desc")
    if not nodes:
        return

    texts = [f"{n['label']}: {n['id']}. {n['desc'] or ''}" for n in nodes]
    try:
        vecs = embeddings.embed_documents(texts)
        for node, vec in zip(nodes, vecs):
            graph.query(
                "MATCH (n {id: $id}) SET n.embedding = $embedding",
                {"id": node["id"], "embedding": vec}
            )
        print(f"  ✓ Stored embeddings for {len(nodes)} nodes")
    except Exception as e:
        print(f"  ✗ Embedding storage error: {e}")

# ── Step 2: Graph Traversal ───────────────────────────────────────────────────
def graph_traversal(entity_ids: list, depth: int = 2) -> list:
    """
    Navigate the directed knowledge graph from seed entities.
    Follows directed edges up to `depth` hops, collecting all
    connected nodes and relationships (supports cycles & multi-edges).
    """
    if not entity_ids:
        return []

    results = graph.query(
        """
        UNWIND $ids AS seed
        MATCH path = (start {id: seed})-[*1..$depth]->(end)
        UNWIND relationships(path) AS r
        RETURN
            startNode(r).id   AS from_node,
            labels(startNode(r))[0] AS from_type,
            type(r)           AS relationship,
            endNode(r).id     AS to_node,
            labels(endNode(r))[0] AS to_type,
            endNode(r).description AS to_desc
        """,
        {"ids": entity_ids, "depth": depth}
    )
    return results

# ── Step 3: Semantic Search via Embeddings ────────────────────────────────────
def semantic_search(question: str, top_k: int = 5) -> list:
    """
    Embed the question and find the most similar nodes using
    cosine similarity against stored node embeddings.
    """
    try:
        q_vec = embeddings.embed_query(question)
        nodes = graph.query(
            "MATCH (n) WHERE n.embedding IS NOT NULL AND n.id IS NOT NULL "
            "RETURN n.id AS id, labels(n)[0] AS label, n.description AS desc, n.embedding AS embedding"
        )
        if not nodes:
            return []

        q_arr = np.array(q_vec)
        scored = []
        for node in nodes:
            n_arr = np.array(node["embedding"])
            # cosine similarity
            sim = float(np.dot(q_arr, n_arr) / (np.linalg.norm(q_arr) * np.linalg.norm(n_arr) + 1e-9))
            scored.append((sim, node))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in scored[:top_k]]

    except Exception as e:
        print(f"  ✗ Semantic search error: {e}")
        # Fallback: return top nodes by label
        return graph.query(
            "MATCH (n) WHERE n.id IS NOT NULL RETURN n.id AS id, labels(n)[0] AS label, n.description AS desc LIMIT $k",
            {"k": top_k}
        )

# ── Full RAG Pipeline ─────────────────────────────────────────────────────────
def rag_pipeline(question: str) -> str:
    """
    Full Graph RAG:
    1. Semantic search  → find relevant seed entities
    2. Graph traversal  → navigate connected entities (directed, multi-edge, cyclic)
    3. LLM synthesis    → generate answer from retrieved context
    """
    # Step 1: Semantic search for seed entities
    print("  [1/3] Semantic search for relevant entities...")
    seed_nodes = semantic_search(question, top_k=5)
    seed_ids = [n["id"] for n in seed_nodes]
    print(f"        Seeds: {seed_ids}")

    # Step 2: Graph traversal from seeds
    print("  [2/3] Graph traversal (directed, multi-edge, cyclic)...")
    traversal_results = graph_traversal(seed_ids, depth=2)

    # Build context from traversal
    context_parts = []
    for r in traversal_results:
        context_parts.append(
            f"({r['from_type']}:{r['from_node']}) -[{r['relationship']}]-> "
            f"({r['to_type']}:{r['to_node']})"
            + (f" | {r['to_desc']}" if r.get('to_desc') else "")
        )

    # Also include seed node descriptions
    for n in seed_nodes:
        if n.get("desc"):
            context_parts.append(f"{n['label']}:{n['id']} — {n['desc']}")

    context = "\n".join(context_parts) if context_parts else "No graph context found."

    # Step 3: LLM answer synthesis
    print("  [3/3] LLM answer synthesis...")
    answer_prompt = PromptTemplate(
        input_variables=["question", "context"],
        template="""You are a knowledge graph assistant. Use the graph context below to answer the question accurately.

Graph Context (entities and relationships):
{context}

Question: {question}

Provide a clear, concise answer based on the graph context. If the context doesn't contain enough information, say so."""
    )

    chain = answer_prompt | llm
    for attempt in range(len(AVAILABLE_MODELS)):
        try:
            response = chain.invoke({"question": question, "context": context})
            break
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "404" in str(e) or "NOT_FOUND" in str(e):
                rotate_model()
                chain = answer_prompt | llm
            else:
                raise

    if hasattr(response, "content"):
        content = response.content
        if isinstance(content, list) and len(content) > 0:
            item = content[0]
            return item["text"] if isinstance(item, dict) and "text" in item else str(item)
        return str(content)
    return str(response)

# ── Schema Display ────────────────────────────────────────────────────────────
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
    print("Pipeline: Semantic Search → Graph Traversal → LLM Answer")
    print("="*50)
    print("Type 'exit' or 'quit' to return to main menu\n")

    schema = get_schema_summary()
    print(f"Graph loaded:\n{schema}\n")
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
            print(f"\n✗ Error: {e}")

# ── Delete Graph Data ────────────────────────────────────────────────────────
def delete_graph_data():
    print("\n" + "="*50)
    print("DELETE ALL GRAPH DATA")
    print("="*50)
    confirm = input("\n⚠ This will delete ALL nodes, relationships and embeddings.\nType 'yes' to confirm: ").strip().lower()
    if confirm != "yes":
        print("Cancelled.")
        return
    try:
        graph.query("MATCH (n) DETACH DELETE n")
        print("✓ All graph data deleted. You can now re-scan your PDFs.")
    except Exception as e:
        print(f"✗ Error deleting data: {e}")

# ── Main Menu ─────────────────────────────────────────────────────────────────
def main_menu():
    create_pdf_folder()

    while True:
        print("\n" + "="*50)
        print("GRAPH RAG SYSTEM")
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
            print("\n✗ Invalid choice.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph RAG System")
    parser.add_argument("--console", action="store_true", help="Run in console mode")
    args = parser.parse_args()

    if args.console:
        main_menu()
    else:
        print("Use --console flag to run the application")
        print("Example: python graph_rag.py --console")
