# Day 29 - Comparing FAISS Distance Thresholds

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load our saved FAISS index
vector_store = FAISS.load_local(
    "day19_faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print("FAISS index loaded successfully!")

# Test question
question = "What are embeddings?"

# Retrieve the top 5 documents
results = vector_store.similarity_search_with_score(
    question,
    k=5
)

# Thresholds to compare
thresholds = [1.0, 1.5, 2.0]

print("\n===== THRESHOLD COMPARISON =====")
print("Question:", question)
print("Documents retrieved:", len(results))

for threshold in thresholds:

    accepted_results = [
        (document, score)
        for document, score in results
        if score < threshold
    ]

    print(f"\n===== THRESHOLD = {threshold} =====")
    print("Documents accepted:", len(accepted_results))
    print("Documents rejected:", len(results) - len(accepted_results))

    for document, score in accepted_results:
        print("FAISS distance:", round(float(score), 3))

        # Day 29 - Test thresholds with different questions

test_questions = [
    "What are embeddings?",
    "What is RAG?",
    "What is the capital of Brazil?"
]

print("\n===== MULTI-QUESTION THRESHOLD TEST =====")

for question in test_questions:

    print("\n------------------------")
    print("Question:", question)

    # Retrieve the top 5 documents
    results = vector_store.similarity_search_with_score(
        question,
        k=5
    )

    # Compare all three thresholds
    for threshold in thresholds:

        accepted_results = [
            (document, score)
            for document, score in results
            if score < threshold
        ]

        print(
            f"Threshold {threshold}: "
            f"{len(accepted_results)} documents accepted"
        )

        # Day 29 - Evaluate threshold accuracy

evaluation_questions = [
    {
        "question": "What are embeddings?",
        "should_answer": True
    },
    {
        "question": "What is RAG?",
        "should_answer": True
    },
    {
        "question": "What is the capital of Brazil?",
        "should_answer": False
    }
]

print("\n===== THRESHOLD ACCURACY REPORT =====")

best_threshold = None
best_accuracy = -1

for threshold in thresholds:

    correct_decisions = 0

    for test in evaluation_questions:

        question = test["question"]
        should_answer = test["should_answer"]

        # Retrieve documents
        results = vector_store.similarity_search_with_score(
            question,
            k=5
        )

        # Apply the current threshold
        accepted_results = [
            (document, score)
            for document, score in results
            if score < threshold
        ]

        # Does the system have documents to use?
        system_can_answer = len(accepted_results) > 0

        # Compare with expected behavior
        if system_can_answer == should_answer:
            correct_decisions += 1

    # Calculate accuracy
    total_tests = len(evaluation_questions)
    accuracy = (correct_decisions / total_tests) * 100

    print(f"\nThreshold: {threshold}")
    print("Correct decisions:", correct_decisions)
    print("Total questions:", total_tests)
    print(f"Accuracy: {accuracy:.1f}%")

    # Select the highest-scoring threshold
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold

print("\n===== BEST THRESHOLD =====")
print("Selected threshold:", best_threshold)
print(f"Acceptance accuracy: {best_accuracy:.1f}%")