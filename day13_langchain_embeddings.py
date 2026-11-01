# Day 13 - LangChain Embeddings

from langchain_huggingface import HuggingFaceEmbeddings

# Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Convert a sentence into an embedding
sentence = "RAG helps AI answer questions using external documents."

vector = embeddings.embed_query(sentence)

print("Embedding model loaded successfully!")
print("Sentence:", sentence)
print("Embedding dimensions:", len(vector))
print("First 5 embedding values:", vector[:5])
# Day 13 - Compare multiple document embeddings

from sentence_transformers import util
import torch

# Sample documents
documents = [
    "RAG retrieves information from external documents.",
    "Embeddings represent text as numerical vectors.",
    "Semantic search finds information based on meaning.",
    "Large Language Models generate human-like text."
]

# Convert documents into embeddings using LangChain
document_vectors = embeddings.embed_documents(documents)

# Ask a question
question = "How does semantic search find relevant information?"

# Convert the question into an embedding
question_vector = embeddings.embed_query(question)

# Calculate cosine similarity
scores = util.cos_sim(
    torch.tensor(question_vector),
    torch.tensor(document_vectors)
)[0]

# Find the most relevant document
best_match_index = scores.argmax().item()

print("\nQuestion:", question)
print("Best matching document:", documents[best_match_index])
print("Similarity score:", round(scores[best_match_index].item(), 3))
# Day 13 - Retrieve top 3 relevant documents

top_results = scores.topk(3)

print("\nTop 3 most relevant documents:")

for rank, (score, index) in enumerate(
    zip(top_results.values, top_results.indices),
    start=1
):
    print(f"\nRank {rank}")
    print("Document:", documents[index.item()])
    print("Similarity score:", round(score.item(), 3))