# Day 12 - LangChain Document Loaders

from langchain_community.document_loaders import TextLoader

# Load our Day 10 text document
loader = TextLoader(
    "day10_knowledge.txt",
    encoding="utf-8"
)

documents = loader.load()

print("Document loaded successfully!")
print("Total documents:", len(documents))

print("\nDocument content:")
print(documents[0].page_content)
# Day 12 - Explore document metadata

document = documents[0]

print("\nDocument metadata:")
print(document.metadata)

print("\nDocument source:")
print(document.metadata.get("source"))

print("\nNumber of characters:")
print(len(document.page_content))
# Day 12 - Load multiple documents

file_names = [
    "rag_basics.txt",
    "embeddings.txt",
    "llm_basics.txt"
]

all_documents = []

for file_name in file_names:
    loader = TextLoader(file_name, encoding="utf-8")
    loaded_documents = loader.load()
    all_documents.extend(loaded_documents)

print("\nTotal documents loaded:", len(all_documents))

for index, doc in enumerate(all_documents, start=1):
    print(f"\nDocument {index}")
    print("Source:", doc.metadata.get("source"))
    print("Content:", doc.page_content)
    # Day 12 - Split documents into chunks

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Create the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

# Split all three documents
split_documents = text_splitter.split_documents(all_documents)

print("\nTotal chunks created:", len(split_documents))

# Display the chunks and their sources
for index, chunk in enumerate(split_documents, start=1):
    print(f"\nChunk {index}")
    print("Source:", chunk.metadata.get("source"))
    print("Content:", chunk.page_content)
    # Day 12 - Explore document metadata

document = documents[0]

print("\nDocument metadata:")
print(document.metadata)

print("\nDocument source:")
print(document.metadata.get("source"))

print("\nNumber of characters:")
print(len(document.page_content))