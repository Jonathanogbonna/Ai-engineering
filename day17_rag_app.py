# Day 17 - Interactive RAG Question-Answering App

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

# Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load the saved FAISS vector store
vector_store = FAISS.load_local(
    "day14_faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Create a retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)

print("Retriever loaded successfully!")

# Load the language model
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

print("FLAN-T5 loaded successfully!")
print("RAG application is ready!")

# Day 17 - Question-answering function

def answer_question(question):

    # Step 1: Retrieve relevant documents
    documents = retriever.invoke(question)

    # Step 2: Combine documents into context
    context = "\n".join(
        document.page_content for document in documents
    )

    # Step 3: Create the prompt
    prompt = f"""
Answer the question using only the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:
"""

    # Step 4: Generate an answer
    result = generator(prompt, max_new_tokens=80)

    return result[0]["generated_text"]


# Test our function
question = "What are embeddings?"

answer = answer_question(question)

print("\nQuestion:", question)
print("AI Answer:", answer)

# Day 17 - Interactive question-answering loop

print("\nWelcome to your RAG Question-Answering App!")
print("Type 'exit' to quit.")

while True:
    question = input("\nAsk a question: ").strip()

    if question.lower() == "exit":
        print("Goodbye!")
        break

    if not question:
        print("Please enter a question.")
        continue

    answer = answer_question(question)

    print("AI Answer:", answer)