# Day 33 - Context-Aware Follow-Up Questions

# Store the current conversation topic
current_topic = None


def rewrite_question(question, topic):

    # If there is no previous topic, keep the question unchanged
    if topic is None:
        return question

    # Check whether the question refers to something earlier
    follow_up_words = [
        "they",
        "them",
        "their",
        "it",
        "its"
    ]

    question_words = question.lower().split()

    # Remove punctuation so "they?" matches "they"
    question_words = [
        word.strip(".,?!")
        for word in question_words
    ]

    # Rewrite the question if it contains a follow-up word
    if any(word in question_words for word in follow_up_words):

        rewritten_question = (
            f"{question} (This question is about {topic}.)"
        )

        return rewritten_question

    return question


# Test the function
print("\n===== FOLLOW-UP QUESTION TEST =====")

current_topic = "embeddings"

questions = [
    "What are embeddings?",
    "Why are they useful?",
    "How do they work?",
    "What is FAISS?"
]

for question in questions:

    rewritten = rewrite_question(question, current_topic)

    print("\nOriginal:", question)
    print("Rewritten:", rewritten)

    # Day 33 - Connect rewritten questions to FAISS

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load the FAISS index we created earlier
vector_store = FAISS.load_local(
    "day19_faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Create a function to search documents
def search_with_context(question, topic):

    # Rewrite the question first
    rewritten_question = rewrite_question(question, topic)

    # Search FAISS using the rewritten question
    results = vector_store.similarity_search_with_score(
        rewritten_question,
        k=3
    )

    print("\nOriginal question:", question)
    print("Search question:", rewritten_question)

    print("\nRetrieved documents:")

    for document, score in results:
        print("\nDocument:", document.page_content)
        print("Distance:", round(float(score), 3))


# Test the context-aware search
print("\n===== CONTEXT-AWARE FAISS SEARCH =====")

search_with_context(
    "Why are they useful?",
    "embeddings"
)

# Day 33 - Generate answers using context-aware RAG

from transformers import pipeline

# Load the language model
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)


def answer_with_context(question, topic):

    # Rewrite the follow-up question
    rewritten_question = rewrite_question(question, topic)

    # Retrieve relevant documents
    results = vector_store.similarity_search_with_score(
        rewritten_question,
        k=3
    )

    # Keep documents below our distance threshold
    relevant_documents = [
        document
        for document, score in results
        if score < 1.5
    ]

    # Return a safe answer if no documents match
    if not relevant_documents:
        return "I don't know."

    # Combine the retrieved documents
    context = "\n".join(
        document.page_content
        for document in relevant_documents
    )

    # Create the prompt
    prompt = f"""
Answer the question using only the document context.
If the answer is not in the context, say "I don't know."

Document context:
{context}

Question: {rewritten_question}

Answer:
"""

    # Generate the answer
    response = generator(
        prompt,
        max_new_tokens=80
    )

    return response[0]["generated_text"]


# Test our context-aware chatbot
print("\n===== CONTEXT-AWARE RAG ANSWER TEST =====")

question = "Why are they useful?"
topic = "embeddings"

answer = answer_with_context(question, topic)

print("\nOriginal question:", question)
print("Previous topic:", topic)
print("Rewritten question:", rewrite_question(question, topic))
print("AI answer:", answer)
# Day 33 - Test multiple follow-up questions

print("\n===== MULTIPLE FOLLOW-UP QUESTIONS TEST =====")

# Each question has a topic from the previous conversation
test_questions = [
    ("What are embeddings?", None),
    ("Why are they useful?", "embeddings"),
    ("How do they help with search?", "embeddings"),
    ("What is FAISS?", None),
    ("Why is it useful?", "FAISS")
]

for question, topic in test_questions:

    rewritten = rewrite_question(question, topic)

    answer = answer_with_context(question, topic)

    print("\n--------------------------------")

    print("Original question:", question)
    print("Previous topic:", topic)
    print("Rewritten question:", rewritten)
    print("AI answer:", answer)

print("\nTotal questions tested:", len(test_questions))

print("\nDAY 33 TEST COMPLETE")