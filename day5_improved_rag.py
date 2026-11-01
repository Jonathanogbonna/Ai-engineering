from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# Step 1: Load our embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 2: Create our knowledge base
documents = [
    "RAG stands for Retrieval-Augmented Generation.",
    "RAG retrieves information from external documents.",
    "Embeddings help AI understand the meaning of text.",
    "Semantic search finds documents based on meaning.",
    "Language models generate answers using context.",
    "Python is a programming language used in AI."
]

# Step 3: Ask a question
question = "How does RAG use documents to answer questions?"

# Step 4: Convert text into embeddings
document_embeddings = model.encode(documents)
question_embedding = model.encode(question)

# Step 5: Calculate similarity scores
scores = util.cos_sim(question_embedding, document_embeddings)[0]

# Step 6: Find the top 3 matching documents
top_results = scores.topk(3)

print("Question:", question)
print("\nTop 3 relevant documents:")

for rank, (score, index) in enumerate(
    zip(top_results.values, top_results.indices), start=1
):
    print(f"\nRank {rank}: {documents[index.item()]}")
    print(f"Similarity score: {score.item():.3f}")
    # Step 7: Combine the top 3 documents into context
retrieved_documents = [
    documents[index.item()]
    for index in top_results.indices
]

context = "\n".join(retrieved_documents)

print("\nCombined context:")
print(context)
# Step 8: Load the language model
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

# Step 9: Build a prompt using the retrieved context
prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question: {question}

Answer:
"""

# Step 10: Generate an answer
result = generator(prompt, max_new_tokens=80)

print("\nAI-generated answer:")
print(result[0]["generated_text"])