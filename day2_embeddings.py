from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Python is a programming language.",
    "RAG helps AI answer questions using external information.",
    "LangChain helps developers build AI applications.",
    "AI agents can use tools to complete tasks."
]

question = "How can artificial intelligence find answers in documents?"

document_embeddings = model.encode(documents)
question_embedding = model.encode(question)

scores = util.cos_sim(question_embedding, document_embeddings)[0]

best_match_index = scores.argmax().item()

print("Question:", question)
print("Best matching document:", documents[best_match_index])
print("Similarity score:", round(scores[best_match_index].item(), 3))