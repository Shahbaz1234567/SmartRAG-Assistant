
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

import numpy as np
import json
from rank_bm25 import BM25Okapi
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Pinecone setup
pc = Pinecone(api_key="Your Pinecone API Key")
index_name = "Your Pinecone Index name"
index = pc.Index(index_name)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# LLM setup
os.environ["GOOGLE_API_KEY"]="GOOGLE_API_KEY"
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    verbose=True, 
    temperature=0, 
    max_tokens=4000, 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Define the prompt template
'''prompt = PromptTemplate(
    input_variables=["query", "groundedContent"],
    template="""You are an assistant tasked with answering a query based only on the provided relevant content. Please follow these instructions carefully:

    1. Answer using only the provided content. Do not add information that is not included in these chunks, and avoid any speculation or hallucination.
    2. Accuracy is key: Only include information explicitly stated in the grounded content.
    3. Grammar check: Correct any grammatical errors in the grounded content.
    4. Clarity and readability: Structure your answer in a clear, coherent, and easy-to-understand way.
    5. Friendly tone: Be warm and approachable in your response.
    6. Keep the answer clear and concise. 
    7. Give final output answer in bullet wise line by line format.
    8. 

    
    Query: {query}
    Grounded Content: {groundedContent}
    """
)'''

prompt = PromptTemplate(
    input_variables=["query", "groundedContent"],
    template="""You are an assistant tasked with answering a query based only on the provided relevant content. Follow these instructions:
 
    1. **Use only the provided content**: Your answer should be based exclusively on the given content. Avoid adding any external information, assumptions, or speculation.
    2. **Be precise and to the point**: Provide a direct and clear response without elaborating. Only address the query as clearly and concisely as possible.
    3. **Accuracy is essential**: Ensure that your answer is exactly aligned with the grounded content. No inferences or guesses.
    4. **Grammar and clarity**: Correct any grammar mistakes in the provided content and ensure the response is easily understandable.
    5. **Friendly tone**: Respond in a helpful, polite, and approachable manner.
    6. **Bullet-point format**: Organize your answer in a bullet-point format for better clarity.
    7. **Keep it short**: Aim for brevity. Eliminate unnecessary details and stick to essential information.
    8. **No repetition**: Avoid repeating points; answer once and clearly.

    Query: {query}
    Grounded Content: {groundedContent}
    """
)


prompt1 = PromptTemplate(
    input_variables=["topic"],
    template="""
        You are tasked with processing user queries and performing Named Entity Recognition (NER) to extract relevant client names and document types. 
        Your goal is to identify specific client names (A, B, C) and document types (MSA, SoW, CR) from the user input.

        User Query: {topic}

        If a client name (A,B,C) is found, return it as 'client_name'. 
        If a document type (MSA, SoW, CR) is found, return it as 'Document_type'.
        If no client name or document type is found, return both fields as 'no'.

        Return the output only in JSON format.

       **Do not** include any extra explanations or text, only the **JSON output**. Make sure to return **only** the JSON structure and no 
        additional words before and after it.
        
    """
)

# Initialize LLMChain
#llm_chain = LLMChain(llm=llm, prompt=prompt)
#llm_chain1 = LLMChain(llm=llm, prompt=prompt1)
llm_chain = prompt | llm
llm_chain1 = prompt1 | llm

def extract_entities1(query: str):
    result = llm_chain1.invoke({"topic": query})
    return result




def Createdict(query):
    
    output = extract_entities1(query)  
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
        print(f"Problematic content: {json_content}")
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
    results = vector_store.similarity_search_with_score(query, k=5, filter=Createdict(query))
    #results = vector_store.similarity_search_with_score(query, k=5)
    #print(results)
    documents = []
    for d in results:
        documents.append(d[0].metadata['Text'])
    #documents = [res.page_content for res, score in results]
    #print(documents)
    tokenized_documents = [doc.split() for doc in documents]
    tokenized_query = query.split()
    
    
    bm25 = BM25Okapi(tokenized_documents)
    bm25_scores = bm25.get_scores(tokenized_query)
    

    sorted_indices = np.argsort(bm25_scores)[::-1]
    reranked_documents = [(results[i], bm25_scores[i]) for i in sorted_indices]
    #for rank, (document, similarity) in enumerate(reranked_documents, start=1):
    # print(f"Rank {rank}: Document - '{document}', Similarity Score - {similarity}")
    grounded_content = ""
    #print(reranked_documents)
    for res,score in reranked_documents:
        #res_str=res[0]
        #start = res_str.find("{") 
        #end = res_str.rfind("}") + 1
        #dictionary_str = res_str[start:end]
        #dictionary_str = dictionary_str.replace("'", '"')
        #metadata = json.loads(dictionary_str)
        #grounded_content = metadata.get('Text', '')+ "\n"
        
        #print(res[0].metadata['Text'])
        grounded_content += res[0].metadata['Text'] + "\n"
        #print(grounded_content)
    return grounded_content


# Function to get a response based on the query and grounded content
def final_output(query:str, grounded_content:str):
    
    result = llm_chain.invoke({"query": query, "groundedContent": grounded_content})
    return result
    

@app.post("/get_answer")
async def get_answer(question: str = Form(...)):
    
    # Retrieve relevant content
    grounded_content = retrieval(question)
    
    # Generate response based on the grounded content
    response = final_output(question, grounded_content) 
    ans=response.content
    print(ans)
    
    # Return the response as JSON
    return JSONResponse(content={"answer": str(ans)})
    #return {"response": grounded_content}



