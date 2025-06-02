import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import numpy as np
import os

load_dotenv()

api_key = os.getenv("YOUR_API_KEY")
if not api_key:
    raise ValueError("API key not found in .env file. Make sure YOUR_API_KEY is set.")

genai.configure(api_key=api_key)

def get_embedding(text: str):
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="semantic_similarity"  
    )
    return response['embedding']

def cosine_sim(vec1, vec2):
    return cosine_similarity(
        np.array(vec1).reshape(1, -1),
        np.array(vec2).reshape(1, -1)
    )[0][0]

def main():
    word1 = "apple"
    word2 = "iphone"

    emb1 = get_embedding(word1)
    emb2 = get_embedding(word2)

    sim = cosine_sim(emb1, emb2)
    print(f"Cosine similarity between '{word1}' and '{word2}': {sim:.4f}")

if __name__ == "__main__":
    main()
