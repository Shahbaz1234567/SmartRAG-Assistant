# AI-Powered Conversational Assistant

## Overview
The AI-Powered Conversational Assistant is designed to provide intelligent, context-aware responses to user queries by leveraging state-of-the-art LLMs (Large Language Models) and retrieval-augmented generation (RAG) techniques. It combines information retrieval, natural language understanding, and conversational AI to deliver accurate, concise, and user-friendly answers.

---

## Features

### 1. Real-Time Query Handling
- Accepts user queries via a clean Streamlit interface.
- Provides answers grounded in relevant content retrieved from a vector database.

### 2. Retrieval-Augmented Generation (RAG)
- Combines Pinecone for vector database storage and Hugging Face models for text embeddings.
- Re-ranks retrieved documents using BM25 for enhanced relevancy.

### 3. Natural Language Processing
- Employs Google Generative AI (Gemini) and Nvidia API for generating responses.
- Provides clarity and readability in generated answers.

### 4. Named Entity Recognition (NER)
- Extracts client names (e.g., JP, Z) and document types (e.g., MSA, SoW) using a tailored LLM-based pipeline.

### 5. Data Parsing and Ingestion
- Extracts structured and unstructured data from documents using **LlamaParse**.
- Processes text, tables, and metadata while preserving original formatting.

### 6. Data Summarization Using Nvidia API
- Summarizes extracted content using Nvidia's API to ensure concise and context-rich outputs.
- Appends metadata and stores summarized data efficiently.

### 7. Data Embedding and Storage using PineconeDB
- Embeds processed data using **HuggingFace's Sentence Transformer** model.
- Stores embedded vectors in a Pinecone vector database for efficient retrieval.

---

## Data Parsing and Ingestion Workflow

### Data Processing Using LlamaParse
The system uses **LlamaParse** to parse documents and extract their contents into structured formats. Key instructions followed during parsing include:

1. **Text Extraction:** Extract text exactly as it appears, preserving headers, bullet points, and formatting.
2. **Table Handling:**
   - Maintain the integrity of multi-page tables as single entities.
   - Preserve column headers consistently across pages.
   - Treat new headers as indicators of new tables.
3. **Metadata Attachment:** Attach client name, document type, and file name metadata to extracted elements.

### Summarization Using Nvidia API
After parsing, the extracted content is summarized using Nvidia's API for better retrieval and analysis. The summarization process involves:
- Maintaining context and avoiding critical information loss.
- Summarizing both textual content and tables.
- Attaching metadata such as client name, document type, and file name.

### Code Workflow for Parsing and Summarization
```python
from llama_parse import LlamaParse
from llama_index.core.node_parser import MarkdownElementNodeParser
from langchain import LLMChain, PromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# Parsing and summarizing the document
path = 'path/to/your/document.pdf'
chunk_size = 1024
chunk_overlap = 200
client_name = 'JP'
doc_type = 'MSA Agreement'

docs, tables = process_and_attach_metadata(path, chunk_size, chunk_overlap, client_name, doc_type, ins1)
```

---

## Data Embedding and Storage
After summarization, the extracted and summarized text and tables are embedded using **HuggingFace's Sentence Transformer** (`all-mpnet-base-v2`) and stored in Pinecone for efficient vector-based similarity search.

### Code Workflow for Data Embedding
```python
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

# Initialize embeddings and Pinecone index
pc = Pinecone(api_key="your-pinecone-api-key")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = PineconeVectorStore(index=pc.Index("your-index-name"), embedding=embeddings)

# Add documents to Pinecone
vector_store.add_documents(documents=combined_documents, ids=uuids)
```

---

## Setup Instructions

### Prerequisites
- Python 3.8+
- Required Python Libraries:
  - `streamlit`
  - `langchain`
  - `pinecone`
  - `sentence-transformers`
  - `google-generativeai`
  - `langchain-nvidia-ai-endpoints`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/ai-conversational-assistant.git
   cd ai-conversational-assistant
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables for API keys:
   ```bash
   export GOOGLE_API_KEY="your-google-api-key"
   export NVIDIA_API_KEY="your-nvidia-api-key"
   export PINECONE_API_KEY="your-pinecone-api-key"
   ```

---

## Usage
1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Enter your query in the text input field.
3. View the AI-generated response and relevant retrieved content.

---

## Future Enhancements
- Add support for multi-modal inputs (e.g., images, audio).
- Enhance summarization capabilities using other LLM APIs.
- Integrate additional LLMs for diverse use cases.

---

## Contributing
We welcome contributions! Please fork the repository and submit a pull request with your proposed changes.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
