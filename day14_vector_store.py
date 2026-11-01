# Day 14 - Introduction to Vector Stores

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Our sample documents
documents = [
    "RAG retrieves information from external documents.",
    "Embeddings convert text into numerical vectors.",
    "Semantic search finds information based on meaning.",
    "Large Language Models generate answers using context."
]

# Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create a FAISS vector store
vector_store = FAISS.from_texts(
    texts=documents,
    embedding=embeddings
)

print("FAISS vector store created successfully!")
print("Total documents:", len(documents))
# Day 14 - Search the FAISS vector store

# Ask a question
question = "How does AI find information based on meaning?"

# Search for the most relevant document
results = vector_store.similarity_search(
    question,
    k=1
)

print("\nQuestion:", question)

print("\nMost relevant document:")
print(results[0].page_content)
# Day 14 - Retrieve the top 3 documents

# Search for the 3 most relevant documents
top_results = vector_store.similarity_search_with_score(
    question,
    k=3
)

print("\nTop 3 most relevant documents:")

for rank, (document, score) in enumerate(top_results, start=1):
    print(f"\nRank {rank}")
    print("Document:", document.page_content)
    print("FAISS distance:", round(float(score), 3))

    # Day 14 - Save the FAISS vector store

vector_store.save_local("day14_faiss_index")

print("\nFAISS vector store saved successfully!")