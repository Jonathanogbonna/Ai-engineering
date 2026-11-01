# Day 37 - Persistent Memory RAG Chatbot

import json
import os

# File for saving Day 37 conversations
history_file = "day37_conversations.json"


# Load previous conversations
def load_conversation_history():

    if os.path.exists(history_file):

        with open(history_file, "r", encoding="utf-8") as file:
            history = json.load(file)

        print("Previous conversations loaded!")
        return history

    else:
        print("No previous conversations found.")
        return []


# Load history when the program starts
conversation_history = load_conversation_history()


# Display the results
print("\n===== PERSISTENT MEMORY SETUP =====")

print("History file:", history_file)

print(
    "Conversations loaded:",
    len(conversation_history)
)

print("\nPERSISTENT MEMORY SETUP COMPLETE")

# Day 37 - Save new conversations
    
def save_conversation(question, answer):
    # Add the new conversation to memory
    conversation_history.append({
        "question": question,
        "answer": answer
    })

    # Save all conversations to the JSON file
    with open(history_file, "w", encoding="utf-8") as file:
        json.dump(conversation_history, file, indent=4)

    print("Conversation saved successfully!")



# Day 37 - Connect persistent memory to RAG

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

# Configure document retrieval
top_k = 3
distance_threshold = 1.5

# Load the embedding model
print("\nLoading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load our saved FAISS index
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

print("\n===== PERSISTENT RAG SETUP TEST =====")

print("Embedding model: Ready")
print("FAISS index: Ready")
print("FLAN-T5: Ready")

print("Conversations in memory:", len(conversation_history))

print("\nPERSISTENT RAG SETUP COMPLETE")
# Day 37 - Interactive RAG chatbot with persistent memory

def chatbot_answer(question):

    # Search for relevant documents
    results = vector_store.similarity_search_with_score(
        question,
        k=top_k
    )

    # Filter documents by distance
    relevant_documents = [
        document
        for document, score in results
        if score < distance_threshold
    ]

    if not relevant_documents:
        return "I don't know."

    # Combine retrieved documents
    context = "\n".join(
        document.page_content
        for document in relevant_documents
    )

    # Include the last two conversations
    history_text = ""

    for message in conversation_history[-2:]:
        history_text += (
            f"User: {message['question']}\n"
            f"AI: {message['answer']}\n"
        )

    # Create the prompt
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

    # Generate the answer
    response = generator(
        prompt,
        max_new_tokens=80
    )

    return response[0]["generated_text"]


# Start the interactive chatbot
print("\n===== AI KNOWLEDGE ASSISTANT =====")
print("Welcome! Ask me questions about AI, RAG, embeddings, and FAISS.")
print("Type 'help' to see available commands.")
print("Type 'exit' to close the chatbot.")
print(f"Saved conversations: {len(conversation_history)}")

while True:

    question = input("\nYou: ").strip()

    if not question:
        print("Please enter a question or type 'help' for commands.")
        continue

    if question.lower() == "exit":
        print("\nGoodbye!")
        break
   
    if question.lower() == "help":
        print("\n===== CHATBOT HELP =====")
        print("help             - Show available commands")
        print("history          - View saved conversations")
        print("search <keyword> - Search conversation history")
        print("stats            - Show chatbot statistics")
        print("exit             - Close the chatbot")
        print("You can also type a question to chat with the AI.")
        continue
    if question.lower() == "stats":
        print("\n===== CHATBOT STATISTICS =====")
        print(f"Total saved messages: {len(conversation_history)}")
        continue


    
    if question.lower() == "clear":
        confirm = input("Delete all saved conversations? (yes/no): ").strip().lower()

        if confirm == "yes":
            conversation_history.clear()

            with open(history_file, "w", encoding="utf-8") as file:
                json.dump(conversation_history, file, indent=4)

            print("Conversation history cleared successfully!")
        else:
            print("Cancelled. Your conversations are safe.")

        continue

    
    if question.lower() == "history":  

        print("\n===== CHAT HISTORY =====")

        if not conversation_history:
            print("No conversations saved yet.")

        else:
            for index, message in enumerate(
                conversation_history, start=1
            ):
                print(f"\nConversation {index}")
                print("Question:", message["question"])
                print("Answer:", message["answer"])

        continue

    if question.lower() == "search":
        print("Please enter a keyword. Example: search Python")
        continue

    if question.lower().startswith("search "):

        keyword = question[7:].strip()
        matches = 0

        print("\n===== SEARCH RESULTS =====")

        for message in conversation_history:

            saved_question = message["question"]
            saved_answer = message["answer"]

            if (
                keyword.lower() in saved_question.lower()
                or keyword.lower() in saved_answer.lower()
            ):
                matches += 1
                print("\nQuestion:", saved_question)
                print("Answer:", saved_answer)

        if matches == 0:
            print("No matching conversations found.")
        else:
            print("\nTotal matches:", matches)

        continue
    if not question:
        print("Please enter a question.")
        continue

    # Generate an AI answer
    answer = chatbot_answer(question)

    print("AI:", answer)

    # Save the question and answer permanently
    save_conversation(question, answer)