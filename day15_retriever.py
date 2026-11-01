# Day 15 - Building a LangChain Retriever

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load the same embedding model used on Day 14
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load the saved FAISS vector store
vector_store = FAISS.load_local(
    "day14_faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print("FAISS vector store loaded successfully!")
# Day 15 - Create a LangChain retriever

# Convert the FAISS vector store into a retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)

# Ask a question
question = "How does AI find information based on meaning?"

# Retrieve relevant documents
results = retriever.invoke(question)

print("\nQuestion:", question)
print("Total documents retrieved:", len(results))

for index, document in enumerate(results, start=1):
    print(f"\nDocument {index}:")
    print(document.page_content)

    # Day 15 - Add source metadata

from langchain_core.documents import Document

# Create documents with source information
source_documents = [
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

# Create a new FAISS vector store with metadata
source_vector_store = FAISS.from_documents(
    source_documents,
    embeddings
)

# Create a retriever
source_retriever = source_vector_store.as_retriever(
    search_kwargs={"k": 2}
)

# Retrieve documents
source_results = source_retriever.invoke(question)

print("\nRetrieved documents with sources:")

for index, document in enumerate(source_results, start=1):
    print(f"\nDocument {index}")
    print("Content:", document.page_content)
    print("Source:", document.metadata.get("source"))