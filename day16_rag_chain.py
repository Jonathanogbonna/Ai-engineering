# Day 16 - Connecting Retrieval to a Language Model

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load our embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load the FAISS vector store saved on Day 14
vector_store = FAISS.load_local(
    "day14_faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Create a retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)

print("LangChain retriever loaded successfully!")

# Ask a question
question = "What is Retrieval-Augmented Generation?"

# Retrieve relevant documents
documents = retriever.invoke(question)

print("\nQuestion:", question)
print("Documents retrieved:", len(documents))

for index, document in enumerate(documents, start=1):
    print(f"\nDocument {index}:")
    print(document.page_content)

    # Day 16 - Generate an AI answer

from transformers import pipeline

# Combine retrieved documents into context
context = "\n".join(
    document.page_content for document in documents
)

print("\nRetrieved context:")
print(context)

# Load FLAN-T5
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

# Create a prompt using the retrieved context
prompt = f"""
Answer the question using only the context below.
If the context does not contain the answer, say "I don't know."

Context:
{context}

Question: {question}

Answer:
"""

# Generate the answer
result = generator(prompt, max_new_tokens=80)

print("\nAI-generated answer:")
print(result[0]["generated_text"])
# Day 16 - Build a reusable LangChain RAG chain

from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate

# Create a prompt template
rag_prompt = PromptTemplate.from_template(
    """Answer the question using only the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""
)

# Retrieve documents and format them as text
def retrieve_context(question):
    retrieved_docs = retriever.invoke(question)
    return {
        "context": "\n".join(
            doc.page_content for doc in retrieved_docs
        ),
        "question": question
    }

# Generate an answer with FLAN-T5
def generate_answer(prompt):
    result = generator(
        prompt.to_string(),
        max_new_tokens=80
    )
    return result[0]["generated_text"]

# Connect the steps into one chain
rag_chain = (
    RunnableLambda(retrieve_context)
    | rag_prompt
    | RunnableLambda(generate_answer)
)

# Test the complete chain
test_question = "What are embeddings?"

answer = rag_chain.invoke(test_question)

print("\nReusable RAG Chain")
print("Question:", test_question)
print("Answer:", answer)