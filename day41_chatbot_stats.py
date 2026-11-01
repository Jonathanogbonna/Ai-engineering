
import json
from pathlib import Path

HISTORY_FILE = Path("day37_conversations.json")


def show_stats():
    if not HISTORY_FILE.exists():
        print("No conversation history found.")
        return

    with open(HISTORY_FILE, "r") as file:
        history = json.load(file)

    print("\n--- Chatbot Statistics ---")
    print(f"Total saved messages: {len(history)}")


if __name__ == "__main__":
    show_stats()
