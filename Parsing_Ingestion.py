# --------                            Data processing using LlamaParse ---------------------------

from llama_parse import LlamaParse
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.llms.gemini import Gemini
import google.generativeai as genai
import google.api_core.exceptions
from llama_index.core import Settings
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from uuid import uuid4
import time
import os
import nest_asyncio
nest_asyncio.apply()
from langchain import LLMChain, PromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from sentence_transformers import SentenceTransformer


ins1 = '''
1. Document should be extracted in the same format:
   - Table of contents should be extracted with exact format without any extra whitespace or formatting.
   - Text as text format.
   - Table as table format.
   - Preserve the order and formatting of all rows exactly as they appear in the document.

2. **Treat Tables as a Single Entity**:
   - If a table spans multiple pages, treat all rows as part of the same table, even if they are on separate pages.

3. **Maintain Column Header Consistency**:
    - If table header is same then column headers should remain the same across all pages and should not be duplicated.
    - If table header is different then column headers will according that new table and no numerical columns headers will be present.

4. **New Headers Indicate a New Table**:
    - If a new set of headers appears on a page, start a new table with the new headers.
    - If no new headers are present, continue the table from the previous page.

5. **Include All Rows**:
    - Ensure that every row from the table is included in the final output, even if it spans multiple pages.
    - Ensure that the last rows are also appended to the single table.

6. Extract the all pages's contents from the document like texts, tables, images, etc without any loss of information.  


'''


def process_and_attach_metadata(file_path, chunk_size, chunk_overlap, client_name, doc_type, pars_ins):
    
    text_prompt_instructions = """
        Generate a detailed summary of the text provided in triple backticks into a concise and coherent overview for effective retrieval. Follow these instructions:
            1. Identify and capture all the topics/headers/sections and key details for the whole text.
            2. Extract all key points from each topic/header/section/note points and use bullet points or short paragraphs to summarize content.
            3. Maintain the context and avoid omitting critical information.
            4. Do not infer or add information not present on the page.
            Respond only with the summary, no additional comment.
            Do not start your message by saying "Here is a summary" or anything like that.
    
    
        text to be summarized:```{content}```
        """
    
    chain = LLMChain(llm=ChatNVIDIA(model="nvidia/llama-3.1-nemotron-70b-instruct",
                 nvidia_api_key = "Nvidia_API_KEY"),  
                 prompt=PromptTemplate.from_template(text_prompt_instructions))



    print(chunk_size)
    documents = LlamaParse(api_key="Llama_api_key", result_type="markdown", parsing_instruction = pars_ins, split_by_page=False, continuous_mode=True, do_not_cache=True).load_data(file_path)

    node_parser = MarkdownElementNodeParser(
        llm=Gemini(model_name="models/gemini-1.5-flash", api_key="Gemini_API_KEY"), num_workers=8
    )

    nodes = node_parser.get_nodes_from_documents(documents, chunk_size, chunk_overlap=chunk_overlap)
    base_nodes, objects = node_parser.get_nodes_and_objects(nodes)

    print("Text elements\n")
    print(base_nodes)
    print(len(base_nodes))
    print("Table elements\n")
    print(objects)

    file_name = os.path.splitext(os.path.basename(file_path))[0]
    print("File Name: " + file_name)

    docs = []
    for i in range(len(base_nodes)):
        base_node = base_nodes[i]
        content = base_node.text 
        summarized = chain.run({'content': content})

        doc = Document(
            page_content=summarized,
            metadata={
                "Client_name": client_name,
                "Document_type": doc_type,
                "File_name": file_name,
                "Text":content

            }
        )
        docs.append(doc)

    print("Text")
    print(docs)

    tables = []
    for i in range(len(objects)):
        obj = objects[i]
        content = obj.obj.text
        summarized = chain.run({'content': content})
        table = Document(
            page_content=summarized,
            metadata={
                "Client_name": client_name,
                "Document_type": doc_type,
                "File_name": file_name,
                "Text":content
            }
        )
        tables.append(table)

    print("Tables")
    print(tables)

    return docs, tables


path = 'C:/Users/shahbaz.ahmad/Documents/CLM/LlamaParse/Nvidia/genai.pdf'

chunk_size = 1024
chunk_overlap = 300
client_name = 'A'
doc_type = 'MSA Agreement'
docs, tables = process_and_attach_metadata(path, chunk_size, chunk_overlap, client_name, doc_type, ins1)

print(docs[1].page_content)
print(tables[0].page_content)


# -------------------------- Data Embedding using HuggingFace Embedding model(Sentence Transformer) + Storing to Pinecone vectorDB using PineconeVectorStore ----------------------------

# # ------- This is for to just cehcking the our embedding model(All-MPNet-base-v2) dimension ----

# model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2') # # Initialize the embedding model and store in model variable

# # Find the length of the embedding model dimension
# sample_text = "This is a sample text."
# sample_vector = model.encode(sample_text).tolist()
# embedding_dimension = len(sample_vector)
# print(f"Embedding model dimension: {embedding_dimension}")



pc = Pinecone(api_key="Your Pinecone API Key")
index_name="Your Pinecone Index NAME"

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=768,  
        metric="cosine",  
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)


index = pc.Index(index_name)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vector_store = PineconeVectorStore(index=index, embedding=embeddings)
combined_documents = docs + tables


uuids = [str(uuid4()) for _ in range(len(combined_documents))]

vector_store.add_documents(documents=combined_documents, ids=uuids)

