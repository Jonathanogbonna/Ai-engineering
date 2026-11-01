# Day 31 - RAG Chatbot with Conversation History

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

# Configure retrieval settings
top_k = 3
distance_threshold = 1.5

# Create an empty conversation history
conversation_history = []

print("\n===== RAG CHATBOT SETUP =====")
print("Top-K:", top_k)
print("Distance threshold:", distance_threshold)
print("Conversation history:", conversation_history)
print("Chatbot initialized successfully!")

# Day 31 - Create the chatbot function

from transformers import pipeline

# Load FLAN-T5
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

def chatbot_answer(question):

    # Retrieve documents from FAISS
    results = vector_store.similarity_search_with_score(
        question,
        k=top_k
    )

    # Filter documents by distance
    accepted_results = [
        (document, score)
        for document, score in results
        if score < distance_threshold
    ]

    # Handle questions without relevant documents
    if not accepted_results:
        return "I don't know."

    # Combine retrieved documents
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

    # Generate the answer
    response = generator(
        prompt,
        max_new_tokens=80
    )

    return response[0]["generated_text"]


# Test the chatbot
print("\n===== CHATBOT TEST =====")

question = "What are embeddings?"
answer = chatbot_answer(question)

print("You:", question)
print("AI:", answer)

# Day 31 - Save conversation history

def chat_with_memory(question):

    # Generate an answer using our RAG chatbot
    answer = chatbot_answer(question)

    # Save the question and answer
    conversation_history.append({
        "question": question,
        "answer": answer
    })

    return answer


print("\n===== CONVERSATION HISTORY TEST =====")

# Ask two questions
questions = [
    "What are embeddings?",
    "What is RAG?"
]

for user_question in questions:

    ai_answer = chat_with_memory(user_question)

    print("\nYou:", user_question)
    print("AI:", ai_answer)

# Display saved conversation history
print("\n===== SAVED CONVERSATION HISTORY =====")

for index, message in enumerate(conversation_history, start=1):

    print(f"\nConversation {index}")
    print("Question:", message["question"])
    print("Answer:", message["answer"])

print("\nTotal conversations saved:", len(conversation_history))

# Day 31 - Use conversation history for follow-up questions

def chat_with_context(question):

    # Start with the user's current question
    search_question = question

    # Add the previous question to help understand follow-ups
    if conversation_history:
        previous_question = conversation_history[-1]["question"]

        search_question = (
            previous_question + "\n" + question
        )

    # Retrieve documents using the expanded question
    results = vector_store.similarity_search_with_score(
        search_question,
        k=top_k
    )

    accepted_results = [
        (document, score)
        for document, score in results
        if score < distance_threshold
    ]

    if not accepted_results:
        answer = "I don't know."

    else:
        context = "\n".join(
            document.page_content
            for document, score in accepted_results
        )

        # Include recent conversation history
        history_text = ""

        for message in conversation_history[-2:]:
            history_text += (
                f"User: {message['question']}\n"
                f"AI: {message['answer']}\n"
            )

        prompt = f"""
Answer the current question using only the document context.
Use conversation history to understand follow-up questions.
If the answer is not supported by the documents, say "I don't know."

Document context:
{context}

Conversation history:
{history_text}

Current question: {question}

Answer:
"""

        response = generator(
            prompt,
            max_new_tokens=80
        )

        answer = response[0]["generated_text"]

    # Save the new conversation
    conversation_history.append({
        "question": question,
        "answer": answer
    })

    return answer


# Start a fresh conversation for this test
conversation_history.clear()

print("\n===== CONTEXT-AWARE CHATBOT TEST =====")

first_question = "What are embeddings?"
first_answer = chat_with_context(first_question)

print("\nYou:", first_question)
print("AI:", first_answer)

follow_up = "Why are they useful?"
follow_up_answer = chat_with_context(follow_up)

print("\nYou:", follow_up)
print("AI:", follow_up_answer)

print("\nMessages saved:", len(conversation_history))