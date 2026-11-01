# Day 28 - Comparing Top-K Retrieval

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load our saved FAISS index
vector_store = FAISS.load_local(
    "day19_faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print("FAISS index loaded successfully!")

# Our test question
question = "What are embeddings?"

# Compare different Top-K values
k_values = [1, 3, 5]

print("\n===== TOP-K RETRIEVAL COMPARISON =====")
print("Question:", question)

for k in k_values:

    print(f"\n===== TOP K = {k} =====")

    results = vector_store.similarity_search_with_score(
        question,
        k=k
    )

    print("Documents retrieved:", len(results))

    for index, (document, score) in enumerate(results, start=1):
        print(f"\nDocument {index}")
        print("FAISS distance:", round(float(score), 3))
        print("Content:", document.page_content)

        # Day 28 - Compare Top-K relevance

distance_threshold = 1.5

print("\n===== TOP-K RELEVANCE SUMMARY =====")

for k in k_values:

    results = vector_store.similarity_search_with_score(
        question,
        k=k
    )

    relevant_results = [
        (document, score)
        for document, score in results
        if score < distance_threshold
    ]

    print(f"\nTop K: {k}")
    print("Documents retrieved:", len(results))
    print("Documents passing threshold:", len(relevant_results))

    if results:
        print(
            "Highest distance:",
            round(float(results[-1][1]), 3)
        )

        # Day 28 - Final Top-K comparison

print("\n===== FINAL TOP-K COMPARISON =====")

best_k = None
best_relevant_count = -1

for k in k_values:

    results = vector_store.similarity_search_with_score(
        question,
        k=k
    )

    relevant_results = [
        (document, score)
        for document, score in results
        if score < distance_threshold
    ]

    relevant_count = len(relevant_results)

    print(f"\nTop K = {k}")
    print("Relevant documents:", relevant_count)
    print("Filtered documents:", len(results) - relevant_count)

    # Prefer more accepted documents.
    # If tied, prefer the smaller k.
    if relevant_count > best_relevant_count:
        best_relevant_count = relevant_count
        best_k = k

print("\n===== EXPERIMENT RESULT =====")
print("Selected Top-K:", best_k)
print("Documents passing threshold:", best_relevant_count)