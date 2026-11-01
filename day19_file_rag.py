# Day 19 - RAG Using Real Text Files

from langchain_community.document_loaders import TextLoader

# Files created earlier in the challenge
file_paths = [
    "rag_basics.txt",
    "embeddings.txt",
    "llm_basics.txt"
]

# Load documents from the files
documents = []

for file_path in file_paths:
    loader = TextLoader(file_path, encoding="utf-8")
    loaded_documents = loader.load()
    documents.extend(loaded_documents)

print("Text files loaded successfully!")
print("Total documents loaded:", len(documents))

# Display document content and source
for index, document in enumerate(documents, start=1):
    print(f"\nDocument {index}")
    print("Source:", document.metadata.get("source"))
    print("Content:", document.page_content)

    # Day 19 - Split documents into chunks

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Create a text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40
)

# Split the loaded documents
chunks = text_splitter.split_documents(documents)

print("\n===== DOCUMENT CHUNKING =====")
print("Original documents:", len(documents))
print("Total chunks created:", len(chunks))

# Display each chunk and its source
for index, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {index}")
    print("Source:", chunk.metadata.get("source"))
    print("Content:", chunk.page_content)

    # Day 19 - Create embeddings and a FAISS vector store

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Convert document chunks into embeddings
# and store them in FAISS
vector_store = FAISS.from_documents(
    chunks,
    embeddings
)

print("\n===== FAISS VECTOR STORE =====")
print("FAISS vector store created successfully!")
print("Total chunks stored:", len(chunks))

# Save the vector store for future use
vector_store.save_local("day19_faiss_index")

print("FAISS vector store saved successfully!")

# Day 19 - Semantic search using real documents

# Create a retriever from the FAISS vector store
retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)

# Ask a question
question = "What are embeddings?"

# Retrieve relevant document chunks
results = retriever.invoke(question)

print("\n===== SEMANTIC SEARCH =====")
print("Question:", question)
print("Chunks retrieved:", len(results))

# Display retrieved chunks and their sources
for index, document in enumerate(results, start=1):
    print(f"\nResult {index}")
    print("Source:", document.metadata.get("source", "Unknown"))
    print("Content:", document.page_content)

    # Day 19 - Generate answers from real documents

from transformers import pipeline

# Load the language model
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

# Combine the retrieved chunks
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

# Generate the answer
response = generator(prompt, max_new_tokens=80)

answer = response[0]["generated_text"]

# Display the final answer
print("\n===== DAY 19 FINAL RAG ANSWER =====")
print("Question:", question)
print("AI Answer:", answer)

# Display the source files
print("\nSources retrieved:")

for source in sorted(set(
    document.metadata.get("source", "Unknown")
    for document in results
)):
    print("-", source)