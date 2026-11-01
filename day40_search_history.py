# Day 40 - Search Chat History

import json
import os

# Your saved chatbot conversations
history_file = "day37_conversations.json"

# Load saved conversations
if os.path.exists(history_file):

    with open(history_file, "r", encoding="utf-8") as file:
        conversation_history = json.load(file)

else:
    conversation_history = []

# Display results
print("\n===== DAY 40 CHAT HISTORY =====")

print("Total conversations:", len(conversation_history))

print("\nCHAT HISTORY LOADED SUCCESSFULLY")

# Day 40 - Search conversations by keyword

def search_history(keyword):

    print("\n===== SEARCH RESULTS =====")

    matches = 0

    for message in conversation_history:

        question = message["question"]
        answer = message["answer"]

        # Search both questions and answers
        if (
            keyword.lower() in question.lower()
            or keyword.lower() in answer.lower()
        ):

            matches += 1

            print("\nQuestion:", question)
            print("Answer:", answer)

    if matches == 0:
        print("No matching conversations found.")

    else:
        print("\nTotal matches:", matches)


# Day 40 - Interactive chat history search

print("\n===== CHAT HISTORY SEARCH =====")
print("Type a keyword to search your conversations.")
print("Type 'exit' to stop.")

while True:

    keyword = input("\nSearch: ").strip()

    if keyword.lower() == "exit":
        print("Goodbye!")
        break

    if not keyword:
        print("Please enter a keyword.")
        continue

    search_history(keyword)