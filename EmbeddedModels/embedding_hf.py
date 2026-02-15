from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os
import numpy as np

load_dotenv(override=True)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ['HF_HOME']='D:/HF_AI_Models'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = 'D:/HF_AI_Models'

# The model name must be a valid sentence-transformers model
# Common options:
# - "sentence-transformers/all-MiniLM-L6-v2" (small, fast)
# - "sentence-transformers/all-mpnet-base-v2" (better quality, default)
# - "sentence-transformers/all-MiniLM-L12-v2" (balanced)
# - "google/embeddinggemma-300m" (if available)

# Using a valid sentence-transformers model

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # Small and fast model
)

documents = ["Cristiano Ronaldo is widely considered one of the greatest football players of all time, known for his incredible athleticism, speed, and scoring prowess. Throughout his illustrious career, he has played for top clubs like Manchester United, Real Madrid, and Juventus, breaking numerous records and winning multiple Ballon d'Or awards. His relentless work ethic, dedication to fitness, and ability to perform under pressure have earned him millions of fans worldwide and established him as a true sporting icon.",
"Lionel Messi is a legendary Argentine forward renowned for his, dribbling skills, precise control, and exceptional playmaking abilities. Spending most of his career at FC Barcelona before moving to Paris Saint-Germain and Inter Miami, he has secured a record number of Ballon d'Or awards and cemented his legacy by leading Argentina to a FIFA World Cup victory in 2022. Known for his humility and extraordinary vision, Messi continues to inspire generations with his artistry on the football field.",
"Virat Kohli is a modern-day icon of cricket, celebrated for his aggressive batting style and exceptional consistency, often regarded as one of the best batsmen in the world. Born in Delhi, he quickly rose to stardom, taking over the captaincy and leading India to historic wins, including a notable Test series victory in Australia. Known as \"King Kohli\" or \"Run Machine,\" his passion for the game, intense dedication to fitness, and ability to chase down targets have broken multiple international records.",
"PV Sindhu is one of India's most accomplished female athletes and a powerhouse in world badminton. She made history by becoming the first Indian woman to win medals at two consecutive Olympic games, securing silver in Rio 2016 and bronze in Tokyo 2020. Known for her immense determination, attacking style, and incredible stamina, she has consistently performed at the highest level in international tournaments, inspiring countless young athletes in India and abroad."]

document_embeddings = embedding.embed_documents(documents)

query = "who is the legendary cricket player?"
query_embedding = embedding.embed_query(query)

#print(f"Embedding dimension: {len(result)}")

result = cosine_similarity([query_embedding], document_embeddings)
print(str(result))

# Cosine similarity scores for each document
scores = cosine_similarity([query_embedding], document_embeddings)[0]

# Create (index, score) pairs
indexed_scores = list(enumerate(scores))
print("Raw (index, score) pairs:", indexed_scores)

# Sort by score (second element) in descending order
sorted_scores = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
print("Sorted (index, score) pairs:", sorted_scores)

index,score = sorted(indexed_scores, key=lambda x: x[1])[-1]

print('query: ',query)
print(documents[index],' with similarity score: ',score)