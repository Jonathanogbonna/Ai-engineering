# Day 35 - Interactive RAG Chatbot

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

# Remember the current conversation topic
current_topic = None

# Configure document retrieval
top_k = 3
distance_threshold = 1.5

# Load the embedding model
print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load the saved FAISS index
print("Loading FAISS index...")

vector_store = FAISS.load_local(
    "day19_faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Load FLAN-T5
print("Loading language model...")

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

print("\n===== CHATBOT SETUP COMPLETE =====")
print("Embedding model: Ready")
print("FAISS index: Ready")
print("FLAN-T5: Ready")
print("Current topic:", current_topic)

# Day 35 - Automatic topic tracking

def detect_topic(question):

    question = question.lower()

    if "embedding" in question:
        return "embeddings"

    elif "faiss" in question:
        return "FAISS"

    elif "rag" in question:
        return "RAG"

    return None


# Rewrite follow-up questions using the current topic
def rewrite_question(question, topic):

    if topic is None:
        return question

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

    if any(word in words for word in follow_up_words):
        return f"{question} (This question is about {topic}.)"

    return question


# Test the functions
print("\n===== TOPIC TRACKING TEST =====")

current_topic = None

test_questions = [
    "What are embeddings?",
    "Why are they useful?",
    "What is FAISS?",
    "Why is it useful?"
]

for question in test_questions:

    detected_topic = detect_topic(question)

    if detected_topic is not None:
        current_topic = detected_topic

    rewritten = rewrite_question(question, current_topic)

    print("\nOriginal:", question)
    print("Current topic:", current_topic)
    print("Rewritten:", rewritten)

print("\nTOPIC TRACKING TEST COMPLETE")

# Day 35 - Generate RAG answers

def chatbot_answer(question):

    global current_topic

    # Detect and remember the topic
    detected_topic = detect_topic(question)

    if detected_topic is not None:
        current_topic = detected_topic

    # Rewrite follow-up questions
    search_question = rewrite_question(
        question,
        current_topic
    )

    # Retrieve relevant documents
    results = vector_store.similarity_search_with_score(
        search_question,
        k=top_k
    )

    # Filter documents by distance
    relevant_documents = [
        document
        for document, score in results
        if score < distance_threshold
    ]

    # Return a fallback if no documents match
    if not relevant_documents:
        return "I don't know."

    # Combine retrieved documents
    context = "\n".join(
        document.page_content
        for document in relevant_documents
    )

    # Create the AI prompt
    prompt = f"""
Answer the question using only the document context.
If the answer is not supported by the context, say "I don't know."

Document context:
{context}

Question: {search_question}

Answer:
"""

    # Generate an answer
    response = generator(
        prompt,
        max_new_tokens=80
    )

    return response[0]["generated_text"]


# Test the chatbot
print("\n===== RAG ANSWER TEST =====")

current_topic = None

questions = [
    "What are embeddings?",
    "Why are they useful?"
]

for question in questions:

    answer = chatbot_answer(question)

    print("\nYou:", question)
    print("AI:", answer)
    print("Current topic:", current_topic)

print("\nRAG ANSWER TEST COMPLETE")

# Day 35 - Interactive RAG chatbot

print("\n===== INTERACTIVE RAG CHATBOT =====")
print("Ask me questions about RAG, embeddings, and FAISS.")
print("Type 'exit' to stop the chatbot.")

# Start a fresh conversation
current_topic = None

while True:

    # Get a question from the user
    question = input("\nYou: ").strip()

    # Stop when the user types exit
    if question.lower() == "exit":
        print("\nGoodbye!")
        break

    # Handle empty questions
    if not question:
        print("Please enter a question.")
        continue

    # Generate an answer
    answer = chatbot_answer(question)

    print("AI:", answer)
    print("Current topic:", current_topic)