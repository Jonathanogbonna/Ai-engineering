# Day 10 - RAG with Real Documents

# Open and read the text document
with open("day10_knowledge.txt", "r", encoding="utf-8") as file:
    document = file.read()

# Display the document
print("Document loaded successfully!")

print("\nDocument content:")
print(document)
# Split the document into smaller chunks

chunks = [
    paragraph.strip()
    for paragraph in document.split("\n\n")
    if paragraph.strip()
]

print("\nTotal chunks:", len(chunks))

for index, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {index}: {chunk}")
    # Day 10 - Create embeddings

from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert document chunks into embeddings
chunk_embeddings = model.encode(chunks)

# Create a question
question = "How does RAG help AI answer questions?"

# Convert the question into an embedding
question_embedding = model.encode(question)

print("\nEmbedding model loaded successfully!")
print("Question:", question)
print("Chunk embeddings shape:", chunk_embeddings.shape)
print("Question embedding shape:", question_embedding.shape)

# Day 10 - Retrieve top 3 relevant chunks

from sentence_transformers import util

# Compare the question with all document chunks
similarity_scores = util.cos_sim(
    question_embedding,
    chunk_embeddings
)[0]

# Select the top 3 most relevant chunks
top_results = similarity_scores.topk(3)

print("\nTop 3 most relevant document chunks:")

for rank, (score, index) in enumerate(
    zip(top_results.values, top_results.indices),
    start=1
):
    print(f"\nRank {rank}")
    print("Chunk:", chunks[index.item()])
    print("Similarity score:", round(score.item(), 3))

    # Day 10 - Generate an answer using retrieved document chunks

from transformers import pipeline

# Combine the top 3 relevant chunks
retrieved_chunks = []

for index in top_results.indices:
    retrieved_chunks.append(chunks[index.item()])

context = "\n".join(retrieved_chunks)

print("\nCombined context:")
print(context)

# Load the language model
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

# Create the prompt
prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question: {question}

Answer:
"""

# Generate the answer
result = generator(prompt, max_new_tokens=80)

print("\nAI-generated answer:")
print(result[0]["generated_text"])