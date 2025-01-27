import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from rank_bm25 import BM25Okapi
import numpy as np
import json
import os

# Pinecone setup
pc = Pinecone(api_key="Your Pinecone API Key")
index_name = "Your Pinecone Index Name"
index = pc.Index(index_name)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# LLM setup
os.environ["GOOGLE_API_KEY"] = "GOOGLE_API_KEY"
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    verbose=True,
    temperature=0,
    max_tokens=4000,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

# Define the prompt template
prompt = PromptTemplate(
    input_variables=["query", "groundedContent"],
    template="""You are an assistant tasked with answering a query based only on the provided relevant content. Please follow these instructions carefully:

1. Answer using only the provided content. Do not add information that is not included in these chunks, and avoid any speculation or hallucination.
2. Accuracy is key: Only include information explicitly stated in the grounded content.
3. Grammar check: Correct any grammatical errors in the grounded content.
4. Clarity and readability: Structure your answer in a clear, coherent, and easy-to-understand way.
5. Friendly tone: Be warm and approachable in your response.
6. Keep the answer clear and concise.
7. Use bullet points in relevant sections to make the answer more readable.

Query: {query}
Grounded Content: {groundedContent}
"""
)

# Named Entity Recognition Prompt
prompt_ner = PromptTemplate(
    input_variables=["topic"],
    template="""
        You are tasked with processing user queries and performing Named Entity Recognition (NER) to extract relevant client names and document types. 
        Your goal is to identify specific client names (A,B,C) and document types (MSA, SoW, CR) from the user input.

        User Query: {topic}

        If a client name (A,B,C) is found, return it as 'client_name'. 
        If a document type (MSA, SoW, CR) is found, return it as 'Document_type'.
        If no client name or document type is found, return both fields as 'no'.

        Return the output only in JSON format.

        Do not include any extra explanations or text, only the JSON output.
    """
)

# Initialize LLMChain
llm_chain = prompt | llm
llm_chain_ner = prompt_ner | llm

# Function to extract entities using NER
def extract_entities(query: str):
    result = llm_chain_ner.invoke({"topic": query})
    return result

def Createdict(query):
    
    output = extract_entities(query)  
    output_text = output.text if hasattr(output, 'text') else str(output)  
 
    # Extract JSON content from the response
    start = output_text.find("{")
    end = output_text.rfind("}")
 
    # If no valid JSON is found, return an empty dictionary
    if start == -1 or end == -1:
        print("No JSON structure found in the response.")
        return {}
 
    # Extract the JSON content
    json_content = output_text[start:end+1]
    # Attempt to parse the JSON content
    try:
        data = json.loads(json_content)
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        st.markdown(f"Problematic content: {json_content}")
        return {}
 
    # Initialize the dictionary to store filtered results
    filtered_data = {}
 
    # Process the extracted data to filter relevant client names, document types, and text
    for key, value in data.items():
        # Ensure value is not 'no', and trim whitespace
        if isinstance(value, str):
            value = value.strip().lower()
 
            # Only include values that are not 'no'
            if value != 'no':
                # Check if the key is 'Client_name' or 'Document_type' and filter accordingly
                if key == "Client_name" and value in ["A", "B", "C"]:
                    filtered_data[key] = value.capitalize()
                elif key == "Document_type" and value in ["MSA", "SoW","CR"]:
                    filtered_data[key] = value.upper()
                
 
    # Return the filtered data with relevant fields (Client_name, Document_type, Text)
    return filtered_data if filtered_data else {}



# Function to retrieve relevant content
def retrieval(query):
    results = vector_store.similarity_search_with_score(query, k=5,filter=Createdict(query))
    documents = [res[0].metadata['Text'] for res in results]

    # BM25 Re-ranking
    tokenized_documents = [doc.split() for doc in documents]
    tokenized_query = query.split()
    bm25 = BM25Okapi(tokenized_documents)
    bm25_scores = bm25.get_scores(tokenized_query)
    sorted_indices = np.argsort(bm25_scores)[::-1]
    reranked_documents = [(results[i], bm25_scores[i]) for i in sorted_indices]

    # for rank, (document, similarity) in enumerate(reranked_documents, start=1):
    #     st.markdown(f"Rank {rank}: Document - '{document}', Similarity Score - {similarity}")

    grounded_content = ""
    for res, score in reranked_documents:
        grounded_content += res[0].metadata['Text'] + "\n"
    return grounded_content

# Function to generate final response
def generate_response(query: str, grounded_content: str):
    result = llm_chain.invoke({"query": query, "groundedContent": grounded_content})
    return result.content

# Streamlit UI Styling
st.markdown(
    """
    <style>
        body {
            background-color: #f4f7f9;
        }
        .header {
            background-color: #4CAF50;
            padding: 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 36px;
        }
        .header p {
            font-size: 18px;
            margin: 0;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Header Section
st.markdown(
    """
    <div class="header">
        <h1>AI-Powered Conversational Assistant 🌟</h1>
        <p>Where innovation meets clarity — your questions, expertly answered.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Input field for user query
query = st.text_input("💬 What's on your mind? Ask away!")

if query:
    with st.spinner("🔍 Finding the most relevant information for you..."):
        grounded_content = retrieval(query)

    with st.spinner("🤖 Generating your personalized response..."):
        answer = generate_response(query, grounded_content)

    # Display the results
    st.subheader("💡 Your Answer:")
    st.markdown(answer)

    st.subheader("📚 Relevant Content Retrieved:")
    st.markdown(grounded_content)

    
