# Day 24 - Building a Safer RAG System

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

# Ask a question
question = "What are embeddings?"

# Retrieve documents with distance scores
results = vector_store.similarity_search_with_score(
    question,
    k=3
)

# Set an experimental distance threshold
distance_threshold = 1.5

# Keep documents below the threshold
filtered_results = [
    (document, score)
    for document, score in results
    if score < distance_threshold
]

print("\n===== RELEVANCE CHECK =====")
print("Question:", question)
print("Documents retrieved:", len(results))
print("Documents accepted:", len(filtered_results))

# Display accepted documents
for index, (document, score) in enumerate(filtered_results, start=1):
    print(f"\nAccepted Document {index}")
    print("Source:", document.metadata.get("source", "Unknown"))
    print("FAISS distance:", round(float(score), 3))

if not filtered_results:
    print("\nNo sufficiently close documents found.")

    # Day 24 - Test an unknown question

unknown_question = "What is the capital of Brazil?"

# Search for relevant documents
unknown_results = vector_store.similarity_search_with_score(
    unknown_question,
    k=3
)

print("\n===== UNKNOWN QUESTION TEST =====")
print("Question:", unknown_question)

# Check the distance of each retrieved document
for index, (document, score) in enumerate(unknown_results, start=1):
    print(f"\nResult {index}")
    print("Source:", document.metadata.get("source", "Unknown"))
    print("FAISS distance:", round(float(score), 3))

# Apply our relevance filter
accepted_results = [
    (document, score)
    for document, score in unknown_results
    if score < distance_threshold
]

print("\nDocuments accepted:", len(accepted_results))

if not accepted_results:
    print("AI Answer: I don't know.")
    print("Reason: No documents passed the relevance filter.")
else:
    print("Some documents passed the distance filter.")
    print("Their content still needs to be checked.")

    # Day 24 - Check whether context supports the question

def check_context_support(question, documents):

    stop_words = {
        "what", "is", "the", "a", "an", "of",
        "how", "does", "do", "are", "in", "to"
    }

    question_words = {
        word.lower().strip("?.!,")
        for word in question.split()
        if word.lower().strip("?.!,") not in stop_words
    }

    context = " ".join(
        document.page_content.lower()
        for document, score in documents
    )

    return any(
        word in context.split()
        for word in question_words
    )


print("\n===== CONTEXT SUPPORT CHECK =====")

# Check a known question
known_question = "What are embeddings?"

known_results = vector_store.similarity_search_with_score(
    known_question,
    k=3
)

known_filtered = [
    (document, score)
    for document, score in known_results
    if score < distance_threshold
]

print("\nQuestion:", known_question)
print(
    "Context support:",
    check_context_support(known_question, known_filtered)
)

# Check the unknown question
print("\nQuestion:", unknown_question)
print(
    "Context support:",
    check_context_support(unknown_question, accepted_results)
)

# Day 24 - Generate answers with safety checks

from transformers import pipeline

# Load FLAN-T5
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)


def safe_rag_answer(question):

    # Retrieve documents with distance scores
    results = vector_store.similarity_search_with_score(
        question,
        k=3
    )

    # Filter documents by distance
    filtered = [
        (document, score)
        for document, score in results
        if score < distance_threshold
    ]

    # Check whether we have usable context
    if not filtered:
        return "I don't know."

    if not check_context_support(question, filtered):
        return "I don't know."

    # Combine accepted documents
    context = "\n".join(
        document.page_content
        for document, score in filtered
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
    response = generator(prompt, max_new_tokens=80)

    return response[0]["generated_text"]


# Test both questions
print("\n===== FINAL SAFE RAG TEST =====")

questions = [
    "What are embeddings?",
    "What is the capital of Brazil?"
]

for question in questions:
    print("\nQuestion:", question)
    print("AI Answer:", safe_rag_answer(question))