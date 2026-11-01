# Day 25 - Evaluating a RAG System

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

# Create our evaluation questions
test_cases = [
    {
        "question": "What is RAG?",
        "expected_topic": "retrieval"
    },
    {
        "question": "What are embeddings?",
        "expected_topic": "vector"
    },
    {
        "question": "What is the capital of Brazil?",
        "expected_topic": "I don't know"
    }
]

print("\n===== RAG EVALUATION DATASET =====")
print("Total test questions:", len(test_cases))

for index, test in enumerate(test_cases, start=1):
    print(f"\nTest {index}")
    print("Question:", test["question"])
    print("Expected topic:", test["expected_topic"])

    # Day 25 - Evaluate document retrieval

print("\n===== RETRIEVAL EVALUATION =====")

# Set our experimental distance threshold
distance_threshold = 1.5

for test in test_cases:

    question = test["question"]

    print("\n------------------------")
    print("Question:", question)
    print("Expected topic:", test["expected_topic"])

    # Retrieve the top 3 documents with scores
    results = vector_store.similarity_search_with_score(
        question,
        k=3
    )

    # Filter documents by distance
    accepted_results = [
        (document, score)
        for document, score in results
        if score < distance_threshold
    ]

    print("Documents retrieved:", len(results))
    print("Documents accepted:", len(accepted_results))

    # Display the accepted documents
    for index, (document, score) in enumerate(
        accepted_results, start=1
    ):
        print(f"\nAccepted Document {index}")
        print("Source:", document.metadata.get("source", "Unknown"))
        print("FAISS distance:", round(float(score), 3))
        print("Content:", document.page_content)

    if not accepted_results:
        print("No documents passed the relevance filter.")

        # Day 25 - Evaluate AI-generated answers

from transformers import pipeline

# Load FLAN-T5
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

print("\n===== ANSWER EVALUATION =====")

for test in test_cases:

    question = test["question"]
    expected_topic = test["expected_topic"]

    # Retrieve documents
    results = vector_store.similarity_search_with_score(
        question,
        k=3
    )

    # Apply relevance filtering
    accepted_results = [
        (document, score)
        for document, score in results
        if score < distance_threshold
    ]

    # Generate an answer only if documents pass
    if not accepted_results:
        answer = "I don't know."
    else:
        context = "\n".join(
            document.page_content
            for document, score in accepted_results
        )

        prompt = f"""
Answer the question using only the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:
"""

        response = generator(prompt, max_new_tokens=80)
        answer = response[0]["generated_text"]

    # Check whether the expected topic appears
    passed = expected_topic.lower() in answer.lower()

    print("\n------------------------")
    print("Question:", question)
    print("Expected topic:", expected_topic)
    print("AI Answer:", answer)
    print("Keyword check:", "PASS" if passed else "FAIL")

    
    # Day 25 - Calculate overall evaluation score

print("\n===== FINAL EVALUATION REPORT =====")

total_tests = len(test_cases)
passed_tests = 0

for test in test_cases:

    question = test["question"]
    expected_topic = test["expected_topic"]

    # Retrieve relevant documents
    results = vector_store.similarity_search_with_score(
        question,
        k=3
    )

    accepted_results = [
        (document, score)
        for document, score in results
        if score < distance_threshold
    ]

    # Generate an answer
    if not accepted_results:
        answer = "I don't know."
    else:
        context = "\n".join(
            document.page_content
            for document, score in accepted_results
        )

        prompt = f"""
Answer the question using only the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:
"""

        response = generator(prompt, max_new_tokens=80)
        answer = response[0]["generated_text"]

    # Check expected keyword
    passed = expected_topic.lower() in answer.lower()

    if passed:
        passed_tests += 1

    print("\nQuestion:", question)
    print("Result:", "PASS" if passed else "FAIL")

# Calculate percentage
score = (passed_tests / total_tests) * 100

print("\n===== EVALUATION SUMMARY =====")
print("Total tests:", total_tests)
print("Passed tests:", passed_tests)
print("Failed tests:", total_tests - passed_tests)
print(f"Keyword-match score: {score:.1f}%")