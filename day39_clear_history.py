# Day 39 - Clear Chat History

import json
import os

# Use a separate test file
# This protects your Day 37 conversations
history_file = "day39_test_history.json"

# Create example conversations
conversation_history = [
    {
        "question": "What is RAG?",
        "answer": "Retrieval-Augmented Generation."
    },
    {
        "question": "What are embeddings?",
        "answer": "Numerical representations of text."
    },
    {
        "question": "What is FAISS?",
        "answer": "A library for similarity search."
    }
]

# Save the example conversations
with open(history_file, "w", encoding="utf-8") as file:
    json.dump(conversation_history, file, indent=4)

print("\n===== DAY 39 TEST HISTORY =====")

print("Test history file:", history_file)
print("Conversations saved:", len(conversation_history))

print("\nTEST HISTORY CREATED SUCCESSFULLY")

# Day 39 - Clear saved conversation history

def clear_history():

    # Remove all conversations from memory
    conversation_history.clear()

    # Save the empty history to the JSON file
    with open(history_file, "w", encoding="utf-8") as file:
        json.dump(conversation_history, file, indent=4)

    print("Chat history cleared successfully!")


# Day 39 - Interactive clear command

print("\n===== INTERACTIVE CLEAR HISTORY =====")

print("Type 'clear' to delete test conversations.")
print("Type 'history' to view test conversations.")
print("Type 'exit' to stop.")

while True:

    command = input("\nYou: ").strip().lower()

    if command == "exit":
        print("Goodbye!")
        break

    elif command == "history":

        if not conversation_history:
            print("No conversations saved yet.")

        else:
            for message in conversation_history:
                print("Question:", message["question"])
                print("Answer:", message["answer"])

    elif command == "clear":

        confirmation = input(
            "Delete all test conversations? (yes/no): "
        ).strip().lower()

        if confirmation == "yes":
            clear_history()

        else:
            print("Clear cancelled.")

    else:
        print("Unknown command.")