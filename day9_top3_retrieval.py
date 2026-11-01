# Day 9 - Top 3 Chunk Retrieval

from sentence_transformers import SentenceTransformer, util

# Sample text chunks
chunks = [
    "RAG stands for Retrieval-Augmented Generation.",
    "RAG uses external information to improve AI answers.",
    "Embeddings convert text into numerical representations.",
    "Semantic search finds information based on meaning.",
    "Cosine similarity measures how similar two embeddings are.",
    "Large Language Models generate answers using context.",
    "Text chunking divides documents into smaller pieces."
]

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model loaded successfully!")
print("Total chunks:", len(chunks))
# Convert all chunks into embeddings
chunk_embeddings = model.encode(chunks)

# Create a question
question = "How does RAG improve AI answers?"

# Convert the question into an embedding
question_embedding = model.encode(question)

print("\nQuestion:", question)
print("Chunk embeddings shape:", chunk_embeddings.shape)
print("Question embedding shape:", question_embedding.shape)
# Calculate similarity between the question and all chunks
similarity_scores = util.cos_sim(
    question_embedding,
    chunk_embeddings
)[0]

# Find the top 3 highest similarity scores
top_results = similarity_scores.topk(3)

print("\nTop 3 most relevant chunks:")

for rank, (score, index) in enumerate(
    zip(top_results.values, top_results.indices),
    start=1
):
    print(f"\nRank {rank}")
    print("Chunk:", chunks[index.item()])
    print("Similarity score:", round(score.item(), 3))

    # Combine the top 3 retrieved chunks into one context

retrieved_chunks = []

for index in top_results.indices:
    retrieved_chunks.append(chunks[index.item()])

context = "\n".join(retrieved_chunks)

print("\nCombined context for the language model:")
print(context)
# Generate an answer using the retrieved context
from transformers import pipeline

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question: {question}

Answer:
"""

result = generator(prompt, max_new_tokens=80)

print("\nAI-generated answer:")
print(result[0]["generated_text"])