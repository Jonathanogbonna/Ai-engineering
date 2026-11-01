# Day 34 - Automatic Conversation Topic Tracking

# Store the current conversation topic
current_topic = None


# Detect the topic of a question
def detect_topic(question):

    question = question.lower()

    if "embedding" in question:
        return "embeddings"

    elif "faiss" in question:
        return "FAISS"

    elif "rag" in question:
        return "RAG"

    return None


# Test automatic topic detection
print("\n===== AUTOMATIC TOPIC DETECTION TEST =====")

questions = [
    "What are embeddings?",
    "Why are they useful?",
    "What is FAISS?",
    "How does RAG work?",
    "Can you explain that?"
]

for question in questions:

    detected_topic = detect_topic(question)

    # Update the topic only when a new topic is detected
    if detected_topic is not None:
        current_topic = detected_topic

    print("\nQuestion:", question)
    print("Detected topic:", detected_topic)
    print("Current topic:", current_topic)

print("\nTOPIC DETECTION TEST COMPLETE")

# Day 34 - Automatically rewrite follow-up questions

def rewrite_with_topic(question, topic):

    # No topic available
    if topic is None:
        return question

    # Words that may refer to a previous topic
    follow_up_words = [
        "it",
        "its",
        "they",
        "them",
        "their",
        "that"
    ]

    words = [
        word.strip(".,?!").lower()
        for word in question.split()
    ]

    # Add the topic if this is a follow-up question
    if any(word in words for word in follow_up_words):
        return f"{question} (This question is about {topic}.)"

    return question


# Test automatic topic tracking and rewriting
print("\n===== AUTOMATIC QUESTION REWRITING TEST =====")

# Start a fresh conversation
current_topic = None

test_questions = [
    "What are embeddings?",
    "Why are they useful?",
    "What is FAISS?",
    "Why is it useful?",
    "How does RAG work?",
    "Can you explain that?"
]

for question in test_questions:

    # Detect a new topic
    detected_topic = detect_topic(question)

    if detected_topic is not None:
        current_topic = detected_topic

    # Rewrite using the automatically tracked topic
    rewritten_question = rewrite_with_topic(
        question,
        current_topic
    )

    print("\nOriginal:", question)
    print("Current topic:", current_topic)
    print("Rewritten:", rewritten_question)

print("\nQUESTION REWRITING TEST COMPLETE")

# Day 34 - Connect automatic topic tracking to FAISS

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


def search_with_topic_tracking(question):

    global current_topic

    # Automatically detect the topic
    detected_topic = detect_topic(question)

    if detected_topic is not None:
        current_topic = detected_topic

    # Rewrite the question using the current topic
    rewritten_question = rewrite_with_topic(
        question,
        current_topic
    )

    # Search FAISS
    results = vector_store.similarity_search_with_score(
        rewritten_question,
        k=3
    )

    print("\nOriginal question:", question)
    print("Current topic:", current_topic)
    print("Search question:", rewritten_question)

    print("\nRetrieved documents:")

    for document, score in results:
        print("Document:", document.page_content)
        print("Distance:", round(float(score), 3))
        print("---")


# Test automatic topic-aware retrieval
print("\n===== AUTOMATIC TOPIC-AWARE FAISS TEST =====")

current_topic = None

search_with_topic_tracking("What are embeddings?")
search_with_topic_tracking("Why are they useful?")

print("\nTOPIC-AWARE FAISS TEST COMPLETE")

# Day 34 - Complete topic-aware RAG chatbot

from transformers import pipeline

# Load the language model
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)


def topic_aware_chatbot(question):

    global current_topic

    # Detect and remember the topic
    detected_topic = detect_topic(question)

    if detected_topic is not None:
        current_topic = detected_topic

    # Rewrite follow-up questions
    rewritten_question = rewrite_with_topic(
        question,
        current_topic
    )

    # Retrieve documents
    results = vector_store.similarity_search_with_score(
        rewritten_question,
        k=3
    )

    # Filter by distance
    relevant_documents = [
        document
        for document, score in results
        if score < 1.5
    ]

    if not relevant_documents:
        answer = "I don't know."

    else:
        context = "\n".join(
            document.page_content
            for document in relevant_documents
        )

        prompt = f"""
Answer the question using only the document context.
If the answer is not supported by the context, say "I don't know."

Document context:
{context}

Question: {rewritten_question}

Answer:
"""

        response = generator(
            prompt,
            max_new_tokens=80
        )

        answer = response[0]["generated_text"]

    print("\nYou:", question)
    print("Current topic:", current_topic)
    print("Rewritten question:", rewritten_question)
    print("AI:", answer)

    return answer


# Test the complete chatbot
print("\n===== COMPLETE TOPIC-AWARE RAG CHATBOT =====")

current_topic = None

test_questions = [
    "What are embeddings?",
    "Why are they useful?",
    "What is FAISS?",
    "Why is it useful?",
    "How does RAG work?",
    "Can you explain that?"
]

for question in test_questions:
    topic_aware_chatbot(question)

print("\nDAY 34 TEST COMPLETE")