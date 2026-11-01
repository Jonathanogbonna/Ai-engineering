# Day 36 - Save Chatbot Conversations

import json

# Name of the file where conversations will be saved
history_file = "day36_conversations.json"

# Day 36 - Load existing conversation history

import os

if os.path.exists(history_file):

    with open(history_file, "r", encoding="utf-8") as file:
        conversation_history = json.load(file)

    print("Previous conversations loaded!")

else:
    conversation_history = []
    print("No previous conversations found.")


# Function to save a conversation
def save_conversation(question, answer):

    conversation_history.append({
        "question": question,
        "answer": answer
    })

    # Save the entire history to a JSON file
    with open(history_file, "w", encoding="utf-8") as file:
        json.dump(conversation_history, file, indent=4)

    print("Conversation saved successfully!")


# Day 36 - Prevent duplicate example conversations

if not conversation_history:

    save_conversation(
        "What is RAG?",
        "Retrieval-Augmented Generation."
    )

    save_conversation(
        "What are embeddings?",
        "Numerical representations of text."
    )

else:
    print("Existing conversations found.")
    print("Skipping example conversations.")
print("\n===== JSON CHAT HISTORY TEST =====")
print("Conversations saved:", len(conversation_history))
print("File created:", history_file)

# Day 36 - Display saved conversation history

print("\n===== SAVED CONVERSATION HISTORY =====")

if not conversation_history:
    print("No conversations saved yet.")

else:
    for index, message in enumerate(conversation_history, start=1):

        print(f"\nConversation {index}")
        print("Question:", message["question"])
        print("Answer:", message["answer"])

print("\nTotal conversations in history:", len(conversation_history))

print("\nCHAT HISTORY DISPLAY COMPLETE")