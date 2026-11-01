# Day 11 - Introduction to LangChain

from langchain_core.prompts import PromptTemplate

# Create a reusable prompt template
prompt_template = PromptTemplate.from_template(
    "Explain {topic} in simple terms for a beginner."
)

# Fill in the template with a topic
prompt = prompt_template.format(topic="Retrieval-Augmented Generation")

# Display the completed prompt
print("My first LangChain prompt:")
print(prompt)

# Day 11 - Create a simple LangChain chain

from langchain_core.runnables import RunnableLambda

# Create a function that formats our prompt
def create_prompt(topic):
    return prompt_template.format(topic=topic)

# Turn the function into a LangChain runnable
formatter = RunnableLambda(create_prompt)

# Connect two components using the pipe operator
chain = formatter | RunnableLambda(
    lambda text: f"Formatted prompt: {text}"
)

# Run the chain
result = chain.invoke("Embeddings")

print("\nMy first LangChain chain:")
print(result)

# Day 11 - Connect LangChain to FLAN-T5

from transformers import pipeline
from langchain_core.output_parsers import StrOutputParser

# Load our language model
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

# Create a function that calls the model
def generate_answer(prompt_value):
    prompt_text = prompt_value.to_string()
    result = generator(prompt_text, max_new_tokens=80)
    return result[0]["generated_text"]

# Build a chain: Prompt -> Model -> Output
ai_chain = (
    prompt_template
    | RunnableLambda(generate_answer)
    | StrOutputParser()
)

# Ask the model a question
answer = ai_chain.invoke({
    "topic": "Retrieval-Augmented Generation"
})

print("\nMy first AI-powered LangChain answer:")
print(answer)