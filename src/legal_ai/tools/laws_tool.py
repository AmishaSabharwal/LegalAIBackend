from crewai.tools import tool
import redis
from google import genai
from typing import List
import numpy as np
from dotenv import load_dotenv
import os
from pydantic import BaseModel

load_dotenv()

MODEL = os.getenv("MODEL")
API_KEY = os.getenv("REDIS_API_KEY")
HOST = os.getenv("REDIS_HOST")
PORT = os.getenv("REDIS_PORT")
PASSWORD = os.getenv("REDIS_PASSWORD")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
 

client = genai.Client(api_key=API_KEY)
redis_client = redis.Redis(host=HOST, port=PORT, decode_responses=False, username="default", password=PASSWORD)

class LawQuery(BaseModel):
    query: str

def connect_redis():
    """Connects to Redis and returns the client."""
    try:
        redis_client.ping()
        print("Connected to Redis successfully!")
        return redis_client
    except redis.exceptions.ConnectionError as e:
        print(f"Could not connect to Redis: {e}")
        return None

def generate_embedding(text: str):
    response = client.models.embed_content(
        model="gemini-embedding-exp-03-07",
        contents=[text]
    )
    return response.embeddings[0].values

def cosine_similarity(vec1, vec2):
    """Calculates cosine similarity between two vectors."""
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0

def search_relevant_sections(redis_client, query: str, top_k: int = 5):
    """Search Redis for top-k relevant sections using cosine similarity."""
    query_embedding = np.array(generate_embedding(query), dtype=np.float32)
    relevant_sections = []

    for key in redis_client.scan_iter("legal:*"):
        section = redis_client.hgetall(key)
        try:
            section_data = {
                k.decode('utf-8'): v.decode('utf-8') if k.decode('utf-8') != "embedding" else v
                for k, v in section.items()
            }
            section_embedding = np.frombuffer(section_data["embedding"], dtype=np.float32)
            similarity = cosine_similarity(query_embedding, section_embedding)
            relevant_sections.append({
                "similarity": similarity,
                "text": section_data["text"],
                "title": section_data["section_title"],
                "section": section_data["section_number"]
            })
        except Exception as e:
            print(f"Error processing section {key}: {e}")
            continue

    sorted_sections = sorted(relevant_sections, key=lambda x: x["similarity"], reverse=True)
    return sorted_sections[:top_k]

@tool("Search sections and laws")
def generate_rag_response(input: LawQuery | str | dict) -> str:
    """Generates an answer using RAG with Gemini based on retrieved sections."""
    query = input.query if isinstance(input, LawQuery) else str(input)
    if not isinstance(query, str):
        print(f"Error: Expected query to be a string, but got {type(query)}")
    r = connect_redis()
    sections = search_relevant_sections(r, query)
    context = "\n\n".join([f"{sec['section']} - {sec['title']}:\n{sec['text']}" for sec in sections])
    prompt = f"""Answer the following legal query using the context provided section and return section number, section title and content applies.

    Query: {query}

    Context:
    {context}

    Answer:"""

    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=[prompt]
    )

    answer = response.text.strip()
    
    return answer