# Day 22 - Understanding FAISS Similarity Scores

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load the saved FAISS index
vector_store = FAISS.load_local(
    "day19_faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print("FAISS index loaded successfully!")

# Ask a question
question = "What are embeddings?"

# Retrieve the top 3 documents with scores
results = vector_store.similarity_search_with_score(
    question,
    k=3
)

print("\n===== SIMILARITY SEARCH RESULTS =====")
print("Question:", question)
print("Documents retrieved:", len(results))

for index, (document, score) in enumerate(results, start=1):
    print(f"\nResult {index}")
    print("Source:", document.metadata.get("source", "Unknown"))
    print("Content:", document.page_content)
    print("FAISS distance:", round(float(score), 3))
    # Day 22 - Compare similarity scores for different questions

questions = [
    "What is RAG?",
    "What are embeddings?",
    "What is a Large Language Model?"
]

print("\n===== COMPARING QUESTIONS =====")

for question in questions:
    print("\nQuestion:", question)

    results = vector_store.similarity_search_with_score(
        question,
        k=2
    )

    for rank, (document, score) in enumerate(results, start=1):
        print(f"\nRank {rank}")
        print("Source:", document.metadata.get("source", "Unknown"))
        print("FAISS distance:", round(float(score), 3))
        print("Content:", document.page_content)

        # Day 22 - Filter documents using FAISS distance

question = "What are embeddings?"

# Retrieve documents with distance scores
results = vector_store.similarity_search_with_score(
    question,
    k=3
)

# Set a distance threshold
distance_threshold = 1.5

print("\n===== FILTERED SEARCH RESULTS =====")
print("Question:", question)
print("Distance threshold:", distance_threshold)

# Keep only documents below the threshold
filtered_results = [
    (document, score)
    for document, score in results
    if score < distance_threshold
]

print("Documents accepted:", len(filtered_results))

for index, (document, score) in enumerate(filtered_results, start=1):
    print(f"\nAccepted Document {index}")
    print("Source:", document.metadata.get("source", "Unknown"))
    print("FAISS distance:", round(float(score), 3))
    print("Content:", document.page_content)

if not filtered_results:
    print("No documents passed the distance threshold.")