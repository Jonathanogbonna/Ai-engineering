
def validate_search(command):
    command = command.strip()

    if command.lower() == "search":
        return "Please enter a keyword. Example: search Python"

    if command.lower().startswith("search "):
        keyword = command[7:].strip()

        if not keyword:
            return "Please enter a keyword. Example: search Python"

        return f"Searching conversation history for: {keyword}"

    return "This is not a search command."


if __name__ == "__main__":
    command = input("You: ")
    print(validate_search(command))
