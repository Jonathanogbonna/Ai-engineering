
def validate_question(question):
    question = question.strip()

    if not question:
        return "Please enter a question or type 'help' for commands."

    return f"You entered: {question}"


if __name__ == "__main__":
    question = input("You: ")
    print(validate_question(question))
