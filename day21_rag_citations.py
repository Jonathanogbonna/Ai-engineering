# Day 21 - Interactive RAG with Source References

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load the FAISS index saved on Day 19
vector_store = FAISS.load_local(
    "day19_faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print("FAISS index loaded successfully!")

# Create a retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)

print("Retriever created successfully!")

# Test retrieving documents and their sources
question = "What are embeddings?"

documents = retriever.invoke(question)

print("\nQuestion:", question)
print("Documents retrieved:", len(documents))

for index, document in enumerate(documents, start=1):
    print(f"\nDocument {index}")
    print("Source:", document.metadata.get("source", "Unknown"))
    print("Content:", document.page_content)

    # Day 21 - Generate an AI answer with sources

from transformers import pipeline

# Load FLAN-T5
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

# Create a reusable function
def answer_with_sources(question):

    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(question)

    # Combine their content
    context = "\n".join(
        doc.page_content for doc in retrieved_docs
    )

    # Create the prompt
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

    # Collect unique source filenames
    sources = sorted(set(
        doc.metadata.get("source", "Unknown")
        for doc in retrieved_docs
    ))

    return answer, sources


# Test the function
test_question = "What are embeddings?"

answer, sources = answer_with_sources(test_question)

print("\n===== ANSWER WITH SOURCES =====")
print("Question:", test_question)
print("AI Answer:", answer)

print("\nRetrieved sources:")
for source in sources:
    print("-", source)

    # Day 21 - Interactive RAG with source references

print("\n===== INTERACTIVE RAG WITH SOURCES =====")
print("Ask questions about your documents.")
print("Type 'exit' to quit.")

while True:
    question = input("\nYour question: ").strip()

    if question.lower() == "exit":
        print("Goodbye!")
        break

    if not question:
        print("Please enter a question.")
        continue

    # Generate an answer and retrieve its sources
    answer, sources = answer_with_sources(question)

    print("\nAI Answer:", answer)

    print("\nRetrieved sources:")
    for source in sources:
        print("-", source)