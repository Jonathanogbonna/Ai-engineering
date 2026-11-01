# Day 27 - RAG Accuracy Evaluation

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

# Create evaluation questions
test_cases = [
    {
        "question": "What is RAG?",
        "should_answer": True
    },
    {
        "question": "What are embeddings?",
        "should_answer": True
    },
    {
        "question": "What is the capital of Brazil?",
        "should_answer": False
    }
]

print("\n===== DAY 27 EVALUATION DATASET =====")
print("Total questions:", len(test_cases))

for test in test_cases:
    print("\nQuestion:", test["question"])
    print("Should answer:", test["should_answer"])

    # Day 27 - Test answer acceptance

distance_threshold = 1.5

print("\n===== ANSWER ACCEPTANCE TEST =====")

for test in test_cases:

    question = test["question"]
    should_answer = test["should_answer"]

    # Retrieve the top 3 documents
    results = vector_store.similarity_search_with_score(
        question,
        k=3
    )

    # Keep documents that pass our distance threshold
    accepted_results = [
        (document, score)
        for document, score in results
        if score < distance_threshold
    ]

    # Decide whether the system can attempt an answer
    system_can_answer = len(accepted_results) > 0

    # Compare with the expected behavior
    passed = system_can_answer == should_answer

    print("\n------------------------")
    print("Question:", question)
    print("Should answer:", should_answer)
    print("Documents accepted:", len(accepted_results))
    print("System can answer:", system_can_answer)
    print("Result:", "PASS" if passed else "FAIL")

    # Day 27 - Calculate answer-acceptance accuracy

print("\n===== FINAL ACCURACY REPORT =====")

total_tests = len(test_cases)
passed_tests = 0

for test in test_cases:

    question = test["question"]
    should_answer = test["should_answer"]

    # Retrieve documents
    results = vector_store.similarity_search_with_score(
        question,
        k=3
    )

    # Apply distance filtering
    accepted_results = [
        (document, score)
        for document, score in results
        if score < distance_threshold
    ]

    # Check whether the system accepts the question
    system_can_answer = len(accepted_results) > 0

    # Compare with expected behavior
    if system_can_answer == should_answer:
        passed_tests += 1

# Calculate accuracy percentage
accuracy = (passed_tests / total_tests) * 100

print("Total questions:", total_tests)
print("Correct decisions:", passed_tests)
print("Incorrect decisions:", total_tests - passed_tests)
print(f"Answer-acceptance accuracy: {accuracy:.1f}%")