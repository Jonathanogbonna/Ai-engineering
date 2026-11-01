# Day 32 - Managing Chatbot Memory

# Set the maximum number of conversations to remember
memory_limit = 3

# Create an empty conversation history
conversation_history = []


# Function to save a conversation
def save_conversation(question, answer):

    conversation_history.append({
        "question": question,
        "answer": answer
    })

    # Remove the oldest conversations if the limit is exceeded
    if len(conversation_history) > memory_limit:
        conversation_history.pop(0)


# Add four example conversations
save_conversation("What is RAG?", "Retrieval-Augmented Generation.")

save_conversation("What are embeddings?", "Numerical vectors.")

save_conversation("Why are they useful?", "They help compare text meaning.")

save_conversation("What is FAISS?", "A library for similarity search.")


# Display the saved conversations
print("\n===== CHATBOT MEMORY TEST =====")

for index, message in enumerate(conversation_history, start=1):

    print(f"\nConversation {index}")
    print("Question:", message["question"])
    print("Answer:", message["answer"])

print("\nMemory limit:", memory_limit)
print("Conversations remembered:", len(conversation_history))

# Day 32 - Connect memory to FAISS

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

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

# Load FLAN-T5
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

# Configure retrieval
top_k = 3
distance_threshold = 1.5

print("\n===== RAG MEMORY SETUP =====")
print("FAISS index loaded successfully!")
print("FLAN-T5 model loaded successfully!")
print("Memory limit:", memory_limit)
print("Conversations currently stored:", len(conversation_history))

# Day 32 - Chatbot with limited memory

def chat_with_limited_memory(question):

    # Use the previous question to help with retrieval
    search_question = question

    if conversation_history:
        previous_question = conversation_history[-1]["question"]

        search_question = previous_question + "\n" + question

    # Retrieve documents
    results = vector_store.similarity_search_with_score(
        search_question,
        k=top_k
    )

    # Filter documents
    accepted_results = [
        (document, score)
        for document, score in results
        if score < distance_threshold
    ]

    if not accepted_results:
        answer = "I don't know."

    else:
        # Build document context
        context = "\n".join(
            document.page_content
            for document, score in accepted_results
        )

        # Build conversation history text
        history_text = ""

        for message in conversation_history:
            history_text += (
                f"User: {message['question']}\n"
                f"AI: {message['answer']}\n"
            )

        # Create the prompt
        prompt = f"""
Answer using only the document context.
Use conversation history to understand follow-up questions.
If the answer is not supported by the documents, say "I don't know."

Document context:
{context}

Conversation history:
{history_text}

Current question: {question}

Answer:
"""

        # Generate an answer
        response = generator(
            prompt,
            max_new_tokens=80
        )

        answer = response[0]["generated_text"]

    # Save the new conversation using our memory limit
    save_conversation(question, answer)

    return answer


# Clear example conversations before testing
conversation_history.clear()

print("\n===== LIMITED MEMORY CHATBOT TEST =====")

questions = [
    "What is RAG?",
    "What are embeddings?",
    "Why are they useful?",
    "What is FAISS?"
]

for question in questions:

    answer = chat_with_limited_memory(question)

    print("\nYou:", question)
    print("AI:", answer)

print("\nConversations remembered:", len(conversation_history))

# Day 32 - Verify the memory limit

print("\n===== FINAL MEMORY VERIFICATION =====")

print("Memory limit:", memory_limit)
print("Conversations stored:", len(conversation_history))

# Display the conversations still remembered
for index, message in enumerate(conversation_history, start=1):

    print(f"\nRemembered Conversation {index}")
    print("Question:", message["question"])
    print("Answer:", message["answer"])

# Check whether the oldest conversation was removed
oldest_question = "What is RAG?"

still_remembered = any(
    message["question"] == oldest_question
    for message in conversation_history
)

print("\nOldest question still remembered:", still_remembered)

if len(conversation_history) == memory_limit and not still_remembered:
    print("MEMORY TEST: PASS")
else:
    print("MEMORY TEST: FAIL")