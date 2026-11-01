# List of documents to read
files = [
    "rag_basics.txt",
    "embeddings.txt",
    "llm_basics.txt"
]

# Read all three documents
documents = []

for filename in files:
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read()
        documents.append(text)

print("Documents loaded successfully!")
print("Total documents:", len(documents))

for filename, text in zip(files, documents):
    print("\nFile:", filename)
    print(text)
    from sentence_transformers import SentenceTransformer, util

# Load our embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Ask a question
question = "How do embeddings help AI understand meaning?"

# Convert documents and question into embeddings
document_embeddings = model.encode(documents)
question_embedding = model.encode(question)

# Compare the question with all three documents
scores = util.cos_sim(question_embedding, document_embeddings)[0]

# Find the best matching document
best_match_index = scores.argmax().item()

print("\nQuestion:", question)
print("Best matching file:", files[best_match_index])
print("Similarity score:", round(scores[best_match_index].item(), 3))
print("Retrieved information:")
print(documents[best_match_index])
from transformers import pipeline

# Load our language model
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

# Use the retrieved document as context
context = documents[best_match_index]

prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question: {question}

Answer:
"""

# Generate an answer
result = generator(prompt, max_new_tokens=80)

print("\nAI-generated answer:")
print(result[0]["generated_text"])