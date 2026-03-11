import os
import argparse
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

load_dotenv()

# Initialize connections
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database=os.getenv("NEO4J_DATABASE")
)

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

transformer = LLMGraphTransformer(llm=llm)

PDF_FOLDER = "pdfs"

# Few-shot examples for cypher query generation
FEW_SHOT_EXAMPLES = """
Example 1:
Question: "Who is Marie Curie?"
Cypher: MATCH (p:Person {{id: "Marie Curie"}}) RETURN p

Example 2:
Question: "What did Marie Curie discover?"
Cypher: MATCH (p:Person {{id: "Marie Curie"}})-[r:DISCOVERED]->(e) RETURN e.id, e

Example 3:
Question: "Show me all relationships for a person"
Cypher: MATCH (p:Person)-[r]->(n) RETURN p.id, type(r), n.id, labels(n)

Example 4:
Question: "What are all the organizations mentioned?"
Cypher: MATCH (o:Organization) RETURN o.id

Example 5:
Question: "Who worked at which organization?"
Cypher: MATCH (p:Person)-[r:WORKED_AT]->(o:Organization) RETURN p.id, o.id
"""

def create_pdf_folder():
    """Create pdfs folder if it doesn't exist"""
    Path(PDF_FOLDER).mkdir(exist_ok=True)
    print(f"✓ PDF folder ready at: {PDF_FOLDER}/")

def load_and_chunk_pdfs():
    """Load all PDFs from folder and chunk them"""
    pdf_files = list(Path(PDF_FOLDER).glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {PDF_FOLDER}/ folder")
        return []
    
    print(f"Found {len(pdf_files)} PDF file(s)")
    
    all_documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file.name}")
        loader = PyPDFLoader(str(pdf_file))
        pages = loader.load()
        
        # Chunk the documents
        chunks = text_splitter.split_documents(pages)
        all_documents.extend(chunks)
        print(f"  → Created {len(chunks)} chunks")
    
    print(f"\nTotal chunks: {len(all_documents)}")
    return all_documents

def scan_and_convert_pdfs():
    """Scan PDFs and convert to graph"""
    print("\n" + "="*50)
    print("SCANNING PDFs AND CONVERTING TO GRAPH")
    print("="*50 + "\n")
    
    documents = load_and_chunk_pdfs()
    
    if not documents:
        return
    
    print("\nConverting documents to graph...")
    print("This may take a few minutes depending on the number of chunks...\n")
    
    # Process in batches to avoid overwhelming the API
    batch_size = 5
    total_batches = (len(documents) + batch_size - 1) // batch_size
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        batch_num = (i // batch_size) + 1
        
        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
        
        try:
            graph_documents = transformer.convert_to_graph_documents(batch)
            graph.add_graph_documents(graph_documents)
            print(f"  ✓ Added {len(graph_documents)} graph documents to Neo4j")
        except Exception as e:
            print(f"  ✗ Error processing batch {batch_num}: {e}")
    
    print("\n✓ PDF scanning and conversion complete!")
    print_schema()

def print_schema():
    """Print the current graph schema"""
    print("\n" + "="*50)
    print("CURRENT GRAPH SCHEMA")
    print("="*50)
    
    try:
        # Get node types
        node_query = "MATCH (n) RETURN DISTINCT labels(n) as NodeType, count(*) as Count"
        nodes = graph.query(node_query)
        
        print("\nNode Types:")
        for node in nodes:
            print(f"  - {node['NodeType']}: {node['Count']} nodes")
        
        # Get relationship types
        rel_query = "MATCH ()-[r]->() RETURN DISTINCT type(r) as RelType, count(*) as Count"
        rels = graph.query(rel_query)
        
        print("\nRelationship Types:")
        for rel in rels:
            print(f"  - {rel['RelType']}: {rel['Count']} relationships")
        
        # Get sample structure
        sample_query = "MATCH (a)-[r]->(b) RETURN labels(a)[0] as From, type(r) as Relationship, labels(b)[0] as To LIMIT 10"
        samples = graph.query(sample_query)
        
        print("\nSample Relationships:")
        for sample in samples:
            print(f"  ({sample['From']})-[{sample['Relationship']}]->({sample['To']})")
            
    except Exception as e:
        print(f"Error fetching schema: {e}")

def get_schema_context():
    """Get schema as context for the chatbot"""
    try:
        node_query = "MATCH (n) RETURN DISTINCT labels(n) as NodeType"
        nodes = graph.query(node_query)
        node_types = [n['NodeType'][0] for n in nodes if n['NodeType']]
        
        rel_query = "MATCH ()-[r]->() RETURN DISTINCT type(r) as RelType"
        rels = graph.query(rel_query)
        rel_types = [r['RelType'] for r in rels]
        
        schema = f"""
Graph Schema:
- Node Types: {', '.join(node_types)}
- Relationship Types: {', '.join(rel_types)}

Sample Structure:
"""
        sample_query = "MATCH (a)-[r]->(b) RETURN labels(a)[0] as From, type(r) as Relationship, labels(b)[0] as To LIMIT 5"
        samples = graph.query(sample_query)
        for sample in samples:
            schema += f"\n({sample['From']})-[{sample['Relationship']}]->({sample['To']})"
        
        return schema
    except:
        return "Schema not available yet. Please scan PDFs first."

def generate_cypher_query(question, schema):
    """Generate Cypher query from natural language question"""
    prompt = f"""You are a Neo4j Cypher query expert. Convert the user's natural language question into a valid Cypher query.

{schema}

{FEW_SHOT_EXAMPLES}

Important Guidelines:
- Use MATCH clauses to find patterns
- Use WHERE clauses for filtering
- Use RETURN to specify what to return
- Use case-insensitive matching with toLower() when needed
- For "who" questions, look for Person nodes
- For "what" questions, look for related entities
- For "where" questions, look for Location nodes
- Always return relevant properties like id, name, or description

User Question: {question}

Generate ONLY the Cypher query, no explanations:"""

    response = llm.invoke(prompt)
    cypher_query = response.content.strip()
    
    # Clean up the query
    if cypher_query.startswith("```"):
        cypher_query = cypher_query.split("```")[1]
        if cypher_query.startswith("cypher"):
            cypher_query = cypher_query[6:]
    
    return cypher_query.strip()

def format_results(results):
    """Format query results for display"""
    if not results:
        return "No results found."
    
    formatted = "\nResults:\n" + "="*50 + "\n"
    
    for i, result in enumerate(results, 1):
        formatted += f"\n{i}. "
        for key, value in result.items():
            if isinstance(value, dict):
                # Node or relationship object
                if 'id' in value:
                    formatted += f"{key}: {value['id']}"
                else:
                    formatted += f"{key}: {value}"
            else:
                formatted += f"{key}: {value}"
            formatted += " | "
        formatted = formatted.rstrip(" | ")
    
    return formatted

def chatbot():
    """Interactive chatbot for querying the graph"""
    print("\n" + "="*50)
    print("CHATBOT - Ask questions about your documents")
    print("="*50)
    print("Type 'exit' or 'quit' to return to main menu\n")
    
    schema = get_schema_context()
    print("Loading graph schema...")
    print(schema)
    print("\n" + "="*50 + "\n")
    
    while True:
        question = input("\nYou: ").strip()
        
        if question.lower() in ['exit', 'quit', 'back']:
            break
        
        if not question:
            continue
        
        try:
            print("\nGenerating query...")
            cypher_query = generate_cypher_query(question, schema)
            print(f"Cypher: {cypher_query}\n")
            
            print("Executing query...")
            results = graph.query(cypher_query)
            
            print(format_results(results))
            
        except Exception as e:
            print(f"\n✗ Error: {e}")
            print("Try rephrasing your question or ask something else.")

def main_menu():
    """Display main menu and handle user choice"""
    create_pdf_folder()
    
    while True:
        print("\n" + "="*50)
        print("GRAPH RAG SYSTEM - Main Menu")
        print("="*50)
        print("\n1. Scan PDFs folder and convert to Graph")
        print("2. Chatbot - Query the Graph")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            scan_and_convert_pdfs()
        elif choice == '2':
            chatbot()
        elif choice == '3':
            print("\nGoodbye!")
            break
        else:
            print("\n✗ Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph RAG System with PDF processing")
    parser.add_argument('--console', action='store_true', help='Run in console mode')
    args = parser.parse_args()
    
    if args.console:
        main_menu()
    else:
        print("Use --console flag to run the application")
        print("Example: python graph_rag.py --console")
