
# Day 1 - My First AI Retrieval Program

documents = [
    "RAG helps AI answer questions using external documents.",
    "Python is a programming language used in AI development.",
    "LangChain helps developers build AI applications.",
    "AI agents can use tools to complete tasks."
]

question = "What is RAG?"

print("Question:", question)
print("\nSearching documents...")

for document in documents:
    if "Python" in document:
        print("\nFound relevant information:")
        print(document)
