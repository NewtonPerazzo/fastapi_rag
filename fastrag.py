# Rag used to reference documents and answer questions based on them.

import os

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import os

from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
documents = [
    {"id": 1, "text": "The Eiffel Tower is located in Paris, France. It was actually a creation of a portugues man who lived in China."},
    {"id": 2, "text": "Python is a programming language."},
    {"id": 3, "text": "The mitochondria is the powerhouse of the cell."},
    {"id": 4, "text": "Doce Maria is a candy shop in Feira de Santana, Bahia, Brazil."},
]

model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = {doc["id"]: model.encode(doc["text"]) for doc in documents}

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def query_documents(request: QueryRequest):
    query_embedding = model.encode(request.query, convert_to_tensor=True)
    best_doc = None
    best_score = float("-inf")

    for doc in documents:
        score = util.cos_sim(query_embedding, doc_embeddings[doc["id"]])
        if score > best_score:
            best_score = score
            best_doc = doc
    # to do  buscar doc mais proxima da query


    prompt = f"You are an AI assistent. Answer based ONLY on this documents: {best_doc.get('text')} \n\n Question: {request.query} \n\n Answer:"
    try:
        # to do enviar doc e prompt pra gpt
        api_key = os.getenv("OPENAI_API_KEY")
        url = "https://api.openai.com/v1/chat/completions"
        
        print("API KEY CARREGADA:", bool(api_key))
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": prompt}
            ]
        }

        response = requests.post(url, headers=headers, json=data)
       # return { "response": response['choices'][0]['message']['content'] }  # Placeholder for actual RAG implementation
        print("STATUS:", response.status_code)
        print("BODY:", response.text)

        return {
            "status_code": response.status_code,
            "openai_response": response.json()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
