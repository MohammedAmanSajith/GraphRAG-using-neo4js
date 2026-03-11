# Graph RAG System with Neo4j and Google Gemini

A powerful Graph-based Retrieval Augmented Generation (RAG) system that converts PDF documents into knowledge graphs using Neo4j and Google Gemini LLM. Query your documents using natural language through an intelligent chatbot interface.

## 🌟 Features

- **PDF to Knowledge Graph**: Automatically converts PDF documents into structured knowledge graphs
- **Intelligent Chunking**: Smart text splitting with overlap for better context preservation
- **Natural Language Queries**: Ask questions in plain English, automatically converted to Cypher queries
- **Schema-Aware RAG**: Uses graph schema and few-shot learning for accurate query generation
- **Interactive Chatbot**: Console-based interface for querying your document knowledge base
- **Neo4j Integration**: Persistent storage in Neo4j graph database

## 📋 Prerequisites

- Python 3.8+
- Neo4j Aura account (free tier available)
- Google Gemini API key (free tier available)

## 🚀 Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd Graphrag
```

2. **Create virtual environment**
```bash
python -m venv graph_venv
source graph_venv/bin/activate  # On Windows: graph_venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

Create a `.env` file in the root directory:
```env
NEO4J_URI=your_neo4j_uri
NEO4J_USERNAME=your_username
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=your_database
GOOGLE_API_KEY=your_gemini_api_key
```

**How to get credentials:**

- **Neo4j Aura**: Sign up at https://console.neo4j.io
  - Create a free instance
  - Copy the connection URI, username, password, and database name

- **Google Gemini API**: Get your key at https://ai.google.dev
  - Create a new API key
  - Free tier includes generous quotas

## 📁 Project Structure

```
Graphrag/
├── graph_rag.py              # Main application (console mode)
├── text_to_graph.ipynb       # Jupyter notebook for experimentation
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (DO NOT COMMIT)
├── .gitignore               # Git ignore file
├── pdfs/                    # Place your PDF files here
└── README.md                # This file
```

## 🎯 Usage

### Console Application

Run the interactive console application:

```bash
python graph_rag.py --console
```

**Main Menu Options:**

1. **Scan PDFs folder and convert to Graph**
   - Place PDF files in the `pdfs/` folder
   - Select option 1 to process and upload to Neo4j
   - Documents are chunked and converted to knowledge graphs

2. **Chatbot - Query the Graph**
   - Ask questions in natural language
   - System generates Cypher queries automatically
   - Get answers from your document knowledge base

3. **Exit**
   - Close the application

### Jupyter Notebook

For experimentation and testing:

```bash
jupyter notebook text_to_graph.ipynb
```

## 💡 Example Usage

### Adding Documents

1. Place PDF files in `pdfs/` folder
2. Run: `python graph_rag.py --console`
3. Select option 1
4. Wait for processing to complete

### Querying Documents

```
You: Who is Marie Curie?
Cypher: MATCH (p:Person {id: "Marie Curie"}) RETURN p

You: What did she discover?
Cypher: MATCH (p:Person {id: "Marie Curie"})-[r:DISCOVERED]->(e) RETURN e.id

You: Show me all organizations mentioned
Cypher: MATCH (o:Organization) RETURN o.id
```

## 🔧 Configuration

### Chunking Parameters

Modify in `graph_rag.py`:
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Adjust chunk size
    chunk_overlap=200,      # Adjust overlap
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
```

### Batch Processing

Adjust batch size for API rate limits:
```python
batch_size = 5  # Process 5 chunks at a time
```

### Model Selection

Change Gemini model:
```python
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",  # or "gemini-2.0-flash"
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)
```

## 🛡️ Security

**IMPORTANT**: Never commit your `.env` file to GitHub!

The `.gitignore` file is configured to exclude:
- `.env` (credentials)
- `__pycache__/` (Python cache)
- `*.pyc` (compiled Python)
- `.ipynb_checkpoints/` (Jupyter checkpoints)
- `graph_venv/` (virtual environment)

## 📊 Graph Schema

The system automatically extracts:

**Node Types:**
- Person
- Organization
- Location
- Concept
- Award
- Element
- Field
- (and more based on your documents)

**Relationship Types:**
- WORKED_AT
- DISCOVERED
- WON
- LOCATED_IN
- (and more based on document content)

## 🐛 Troubleshooting

### Neo4j Connection Issues
- Verify your instance is running at https://console.neo4j.io
- Check credentials in `.env` file
- Wait 60 seconds after creating a new instance

### API Rate Limits
- Free tier has rate limits
- Reduce `batch_size` if hitting limits
- Wait between requests if quota exceeded

### PDF Processing Errors
- Ensure PDFs are text-based (not scanned images)
- Check PDF file permissions
- Verify `pdfs/` folder exists

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) - LLM framework
- [Neo4j](https://neo4j.com/) - Graph database
- [Google Gemini](https://ai.google.dev/) - LLM API

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Note**: This is a demonstration project. For production use, implement proper error handling, logging, and security measures.
