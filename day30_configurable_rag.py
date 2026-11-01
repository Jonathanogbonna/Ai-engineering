# Day 30 - Configurable RAG System

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 2: Load the saved FAISS index
vector_store = FAISS.load_local(
    "day19_faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print("FAISS index loaded successfully!")

# Step 3: Configure RAG settings
top_k = 3
distance_threshold = 1.5

print("\n===== RAG CONFIGURATION =====")
print("Top-K:", top_k)
print("Distance threshold:", distance_threshold)

# Step 4: Define a test question
question = "What are embeddings?"

print("\nTest question:", question)

# Day 30 - Retrieve and filter documents

print("\n===== DOCUMENT RETRIEVAL =====")

# Retrieve documents using our Top-K setting
results = vector_store.similarity_search_with_score(
    question,
    k=top_k
)

print("Documents retrieved:", len(results))

# Filter documents using our distance threshold
accepted_results = [
    (document, score)
    for document, score in results
    if score < distance_threshold
]

print("Documents accepted:", len(accepted_results))
print("Documents rejected:", len(results) - len(accepted_results))

# Display accepted documents
for index, (document, score) in enumerate(
    accepted_results,
    start=1
):
    print(f"\nAccepted Document {index}")
    print("FAISS distance:", round(float(score), 3))
    print("Content:", document.page_content)

if not accepted_results:
    print("\nNo documents passed the relevance filter.")

    # Day 30 - Generate an AI answer

from transformers import pipeline

# Load FLAN-T5
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

print("\n===== AI GENERATED ANSWER =====")

# Check whether any documents passed the filter
if not accepted_results:
    answer = "I don't know."

else:
    # Combine accepted documents into context
    context = "\n".join(
        document.page_content
        for document, score in accepted_results
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

    # Generate an answer
    response = generator(
        prompt,
        max_new_tokens=80
    )

    answer = response[0]["generated_text"]

print("Question:", question)
print("AI Answer:", answer)

# Day 30 - Reusable configurable RAG function

def answer_question(question, top_k=3, distance_threshold=1.5):

    # Retrieve documents
    results = vector_store.similarity_search_with_score(
        question,
        k=top_k
    )

    # Filter documents
    accepted_results = [
        (document, score)
        for document, score in results
        if score < distance_threshold
    ]

    # Return a fallback if no documents are accepted
    if not accepted_results:
        return "I don't know."

    # Build context
    context = "\n".join(
        document.page_content
        for document, score in accepted_results
    )

    # Create prompt
    prompt = f"""
Answer the question using only the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:
"""

    # Generate answer
    response = generator(
        prompt,
        max_new_tokens=80
    )

    return response[0]["generated_text"]


# Test the reusable function
print("\n===== REUSABLE RAG FUNCTION TEST =====")

questions = [
    "What is RAG?",
    "What are embeddings?",
    "What is the capital of Brazil?"
]

for test_question in questions:

    answer = answer_question(
        test_question,
        top_k=3,
        distance_threshold=1.5
    )

    print("\nQuestion:", test_question)
    print("AI Answer:", answer)