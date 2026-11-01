# Day 20 - Reusable Document Q&A Application

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load the same embedding model used on Day 19
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load the FAISS index saved on Day 19
vector_store = FAISS.load_local(
    "day19_faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print("Saved FAISS index loaded successfully!")

# Create a retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)

print("Document retriever created successfully!")

# Day 20 - Document question-answering function

from transformers import pipeline

# Load FLAN-T5
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

print("FLAN-T5 loaded successfully!")


# Create a reusable function
def answer_question(question):

    # Retrieve relevant document chunks
    documents = retriever.invoke(question)

    # Combine the retrieved text
    context = "\n".join(
        document.page_content for document in documents
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

    return answer


# Test the function
question = "What are embeddings?"

answer = answer_question(question)

print("\n===== DOCUMENT Q&A =====")
print("Question:", question)
print("AI Answer:", answer)

# Day 20 - Interactive Document Q&A

print("\n===== INTERACTIVE DOCUMENT Q&A =====")
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

    answer = answer_question(question)

    print("AI Answer:", answer)