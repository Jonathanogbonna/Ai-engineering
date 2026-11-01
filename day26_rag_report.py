# Day 26 - RAG Evaluation Report

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

# Evaluation questions
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

print("\n===== DAY 26 EVALUATION DATA =====")
print("Total questions:", len(test_cases))

for test in test_cases:
    print("\nQuestion:", test["question"])
    print("Expected topic:", test["expected_topic"])

    # Day 26 - Generate answers and collect results

from transformers import pipeline

# Load FLAN-T5
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

distance_threshold = 1.5

# Store evaluation results
evaluation_results = []

print("\n===== GENERATING EVALUATION RESULTS =====")

for test in test_cases:

    question = test["question"]
    expected_topic = test["expected_topic"]

    # Retrieve documents from FAISS
    results = vector_store.similarity_search_with_score(
        question,
        k=3
    )

    # Keep documents below our distance threshold
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

    # Basic keyword evaluation
    passed = expected_topic.lower() in answer.lower()
    result = "PASS" if passed else "FAIL"

    # Save the result in our list
    evaluation_results.append({
        "question": question,
        "expected_topic": expected_topic,
        "ai_answer": answer,
        "result": result
    })

    print("\nQuestion:", question)
    print("AI Answer:", answer)
    print("Result:", result)

print("\nTotal results collected:", len(evaluation_results))

# Day 26 - Export evaluation results to CSV

import csv

filename = "day26_evaluation_report.csv"

with open(filename, "w", newline="", encoding="utf-8") as file:

    fieldnames = [
        "question",
        "expected_topic",
        "ai_answer",
        "result"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    # Write column headings
    writer.writeheader()

    # Write evaluation results
    writer.writerows(evaluation_results)

print("\n===== CSV REPORT CREATED =====")
print("File:", filename)
print("Total rows:", len(evaluation_results))