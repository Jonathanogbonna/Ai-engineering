# Day 18 - RAG with Source Citations

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Create documents with source information
documents = [
    Document(
        page_content="RAG retrieves information from external documents.",
        metadata={"source": "rag_basics.txt"}
    ),
    Document(
        page_content="Embeddings convert text into numerical vectors.",
        metadata={"source": "embeddings.txt"}
    ),
    Document(
        page_content="Semantic search finds information based on meaning.",
        metadata={"source": "embeddings.txt"}
    ),
    Document(
        page_content="Large Language Models generate answers using context.",
        metadata={"source": "llm_basics.txt"}
    )
]

# Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create a FAISS vector store with source metadata
vector_store = FAISS.from_documents(
    documents,
    embeddings
)

print("FAISS vector store created with source metadata!")

# Create a retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)

print("Retriever created successfully!")

# Day 18 - Retrieve documents with source information

# Ask a question
question = "How does AI find information based on meaning?"

# Retrieve the two most relevant documents
results = retriever.invoke(question)

print("\nQuestion:", question)
print("Documents retrieved:", len(results))

# Display the text and source of each document
for index, document in enumerate(results, start=1):
    print(f"\nDocument {index}")
    print("Content:", document.page_content)
    print("Source:", document.metadata.get("source", "Unknown"))

    # Day 18 - Generate an answer with source citations

from transformers import pipeline

# Load FLAN-T5
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

# Combine retrieved documents into context
context = "\n".join(
    document.page_content for document in results
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

# Generate an answer
response = generator(prompt, max_new_tokens=80)

answer = response[0]["generated_text"]

print("\n===== FINAL RAG ANSWER =====")
print("Question:", question)
print("AI Answer:", answer)

# Display supporting sources
print("\nSources used:")

for document in results:
    print("-", document.metadata.get("source", "Unknown"))