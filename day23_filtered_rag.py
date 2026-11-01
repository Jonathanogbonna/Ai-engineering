# Day 23 - RAG with Relevance Filtering

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load the saved FAISS index from Day 19
vector_store = FAISS.load_local(
    "day19_faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print("FAISS index loaded successfully!")

# Ask a question
question = "What are embeddings?"

# Retrieve the top 3 documents with distance scores
results = vector_store.similarity_search_with_score(
    question,
    k=3
)

print("\n===== RETRIEVAL RESULTS =====")
print("Question:", question)
print("Documents retrieved:", len(results))

for index, (document, score) in enumerate(results, start=1):
    print(f"\nDocument {index}")
    print("Source:", document.metadata.get("source", "Unknown"))
    print("FAISS distance:", round(float(score), 3))
    print("Content:", document.page_content)

    # Day 23 - Filter documents by FAISS distance

# Set a distance threshold
distance_threshold = 1.5

# Keep only documents below the threshold
filtered_results = [
    (document, score)
    for document, score in results
    if score < distance_threshold
]

print("\n===== FILTERED DOCUMENTS =====")
print("Distance threshold:", distance_threshold)
print("Documents accepted:", len(filtered_results))

# Display accepted documents
for index, (document, score) in enumerate(filtered_results, start=1):
    print(f"\nAccepted Document {index}")
    print("Source:", document.metadata.get("source", "Unknown"))
    print("FAISS distance:", round(float(score), 3))
    print("Content:", document.page_content)

# Handle cases where no documents pass the filter
if not filtered_results:
    print("\nNo sufficiently close documents found.")

    # Day 23 - Generate an answer using filtered documents

from transformers import pipeline

# Load FLAN-T5
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

print("\n===== FILTERED RAG ANSWER =====")
print("Question:", question)

# Check whether any documents passed the filter
if filtered_results:

    # Combine only the accepted documents
    context = "\n".join(
        document.page_content
        for document, score in filtered_results
    )

    # Create a prompt
    prompt = f"""
Answer the question using only the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:
"""

    # Generate the answer
    response = generator(prompt, max_new_tokens=80)

    answer = response[0]["generated_text"]

    print("AI Answer:", answer)

    print("\nRetrieved sources:")
    for source in sorted(set(
        document.metadata.get("source", "Unknown")
        for document, score in filtered_results
    )):
        print("-", source)

else:
    print("AI Answer: I don't know.")
    print("Reason: No documents passed the relevance filter.")