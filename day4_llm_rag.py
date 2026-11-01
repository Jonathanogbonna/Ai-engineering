from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# Load our two AI models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

# Our knowledge base
documents = [
    "RAG stands for Retrieval-Augmented Generation.",
    "RAG retrieves relevant information before generating an answer.",
    "Embeddings represent the meaning of text as numbers.",
    "Python is a programming language used in AI development."
]

# Ask a question
question = "What does RAG stand for?"

# Retrieve the most relevant information
document_embeddings = embedding_model.encode(documents)
question_embedding = embedding_model.encode(question)

scores = util.cos_sim(question_embedding, document_embeddings)[0]
best_match_index = scores.argmax().item()

context = documents[best_match_index]

# Give the retrieved information to the language model
prompt = f"""
Answer the question using only the context below.

Context: {context}

Question: {question}

Answer:
"""

result = generator(prompt, max_new_tokens=60)

print("Question:", question)
print("Retrieved context:", context)
print("AI-generated answer:", result[0]["generated_text"])