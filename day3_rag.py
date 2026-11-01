from sentence_transformers import SentenceTransformer, util

# Step 1: Load our embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 2: Create a small knowledge base
documents = [
    "RAG stands for Retrieval-Augmented Generation.",
    "RAG retrieves relevant information before generating an answer.",
    "Embeddings represent the meaning of text as numbers.",
    "Python is a popular programming language for AI."
]

# Step 3: Ask a question
question = "What does RAG mean?"

# Step 4: Convert text into embeddings
document_embeddings = model.encode(documents)
question_embedding = model.encode(question)

# Step 5: Find the most relevant document
scores = util.cos_sim(question_embedding, document_embeddings)[0]
best_match_index = scores.argmax().item()

retrieved_document = documents[best_match_index]

print("Question:", question)
print("Retrieved information:", retrieved_document)

# Step 6: Augment the question with retrieved information
context = retrieved_document

# Step 7: Generate a simple answer from the context
def generate_answer(question, context):
    return f"Based on the retrieved information: {context}"

answer = generate_answer(question, context)

print("\nGenerated answer:", answer)