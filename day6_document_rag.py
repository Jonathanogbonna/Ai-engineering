# Open and read our knowledge document
with open("ai_knowledge.txt", "r", encoding="utf-8") as file:
    document_text = file.read()

# Display the document content
print("Document loaded successfully!")
print("\nDocument content:")
print(document_text)

from sentence_transformers import SentenceTransformer, util
# Split the document into separate sentences
documents = document_text.strip().splitlines()

# Load our embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Ask a question
question = "How does RAG help AI answer questions?"

# Convert the document and question into embeddings
document_embeddings = model.encode(documents)
question_embedding = model.encode(question)

# Compare their meanings
scores = util.cos_sim(question_embedding, document_embeddings)[0]

# Find the most relevant sentence
best_match_index = scores.argmax().item()

print("\nQuestion:", question)
print("Most relevant information:", documents[best_match_index])
print("Similarity score:", round(scores[best_match_index].item(), 3))

from transformers import pipeline

# Load the language model
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

# Get the most relevant information
context = documents[best_match_index]

# Create a prompt using the retrieved information
prompt = f"""
Answer the question using only the context below.

Context: {context}

Question: {question}

Answer:
"""

# Generate the answer
result = generator(prompt, max_new_tokens=80)

print("\nAI-generated answer:")
print(result[0]["generated_text"])