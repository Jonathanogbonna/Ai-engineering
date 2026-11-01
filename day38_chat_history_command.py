# Day 38 - Chat History Command

import json
import os

# Use the conversation file created on Day 37
history_file = "day37_conversations.json"


# Function to load saved conversations
def load_history():

    if os.path.exists(history_file):

        with open(history_file, "r", encoding="utf-8") as file:
            history = json.load(file)

        print("Previous conversations loaded!")
        return history

    else:
        print("No saved conversations found.")
        return []


# Load conversation history
conversation_history = load_history()

# Test the loaded history
print("\n===== DAY 38 HISTORY TEST =====")
print("History file:", history_file)
print("Conversations loaded:", len(conversation_history))

print("\nHISTORY LOADING COMPLETE")

# Day 38 - Display saved chat history

def show_history():

    print("\n===== CHAT HISTORY =====")

    if not conversation_history:
        print("No conversations saved yet.")
        return

    for index, message in enumerate(conversation_history, start=1):

        print(f"\nConversation {index}")
        print("Question:", message["question"])
        print("Answer:", message["answer"])

    print("\nTotal conversations:", len(conversation_history))


# Test the history display function
print("\n===== HISTORY DISPLAY TEST =====")

show_history()

print("\nHISTORY DISPLAY COMPLETE")

# Day 38 - Interactive history command

print("\n===== INTERACTIVE CHAT HISTORY =====")
print("Type 'history' to view saved conversations.")
print("Type 'exit' to stop.")

while True:

    command = input("\nYou: ").strip().lower()

    if command == "exit":
        print("Goodbye!")
        break

    elif command == "history":
        show_history()

    elif not command:
        print("Please enter a command.")

    else:
        print("Unknown command. Type 'history' or 'exit'.")