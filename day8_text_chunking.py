# Day 8 - Understanding Text Chunking

# Our sample document
document = """
RAG stands for Retrieval-Augmented Generation.
RAG helps AI answer questions using external information.
Embeddings convert text into numerical representations.
Semantic search finds information based on meaning.
Large Language Models generate answers using context.
Text chunking divides large documents into smaller pieces.
"""

# Split the document into individual sentences
chunks = document.strip().splitlines()

# Display the chunks
print("Total chunks:", len(chunks))

for index, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {index}: {chunk}")

    # Group 2 sentences into each larger chunk
chunk_size = 2

larger_chunks = []

for i in range(0, len(chunks), chunk_size):
    larger_chunk = " ".join(chunks[i:i + chunk_size])
    larger_chunks.append(larger_chunk)

print("\nTotal larger chunks:", len(larger_chunks))

for index, chunk in enumerate(larger_chunks, start=1):
    print(f"\nLarger Chunk {index}: {chunk}")

    from sentence_transformers import SentenceTransformer, util

# Load our embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Ask a question
question = "How does AI find information based on meaning?"

# Convert our larger chunks into embeddings
chunk_embeddings = model.encode(larger_chunks)
question_embedding = model.encode(question)

# Compare the question with each chunk
scores = util.cos_sim(question_embedding, chunk_embeddings)[0]

# Find the most relevant chunk
best_match_index = scores.argmax().item()

print("\nQuestion:", question)
print("Best matching chunk:", larger_chunks[best_match_index])
print("Similarity score:", round(scores[best_match_index].item(), 3))
# Day 8 - Overlapping chunks

chunk_size = 3
overlap = 1

overlapping_chunks = []

step = chunk_size - overlap

for i in range(0, len(chunks), step):
    chunk = chunks[i:i + chunk_size]

    if len(chunk) == chunk_size:
        overlapping_chunks.append(" ".join(chunk))

print("\nOverlapping chunks:")

for index, chunk in enumerate(overlapping_chunks, start=1):
    print(f"\nChunk {index}: {chunk}")