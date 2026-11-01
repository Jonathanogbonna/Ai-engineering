
# AI Knowledge Assistant — RAG Chatbot

## Project Overview

This project is a Python-based AI chatbot built using Retrieval-Augmented Generation (RAG).

The chatbot retrieves relevant information from documents and uses an AI language model to answer questions.

## Technologies Used

- Python
- LangChain
- FAISS vector database
- Hugging Face embeddings
- FLAN-T5 language model

## Features

- Ask questions about AI topics
- Retrieve relevant document information
- Save conversation history to a JSON file
- View previous conversations using `history`
- Search saved conversations using `search <keyword>`
- Display conversation statistics using `stats`
- View available commands using `help`
- Clear conversation history with confirmation using `clear`
- Exit safely using `exit`

## Current Status

The chatbot is functional and has been tested. Further improvements are planned for answer quality and documentation.

## How to Run the Chatbot

1. Open the project folder in VS Code.
2. Open the Terminal.
3. Activate the Python virtual environment.

   On macOS:

   ```bash
   source .venv/bin/activate
   ```

4. Start the chatbot:

   ```bash
   python3 day37_persistent_rag.py
   ```

5. Wait for the AI Knowledge Assistant welcome message.
6. Type a question or use one of the available commands.

## Example Questions

- What is RAG?
- What are embeddings?
- What is FAISS?

## Available Commands

| Command | Description |
| --- | --- |
| `help` | Show available commands |
| `history` | View saved conversations |
| `search RAG` | Search saved conversations for RAG |
| `stats` | Show total saved messages |
| `clear` | Clear history after confirmation |
| `exit` | Close the chatbot |

## Installation

### Requirements

- Python 3.13
- pip
- Internet connection for downloading AI models

### Setup

1. Download or clone this project.
2. Open the project folder in a Terminal.
3. Create a virtual environment:

   ```bash
   python3 -m venv .venv
   ```

4. Activate the virtual environment on macOS:

   ```bash
   source .venv/bin/activate
   ```

5. Install the Python dependencies:

   ```bash
   python3 -m pip install -r requirements.txt
   ```

6. Run the chatbot:

   ```bash
   python3 day37_persistent_rag.py
   ```

Note: The first run may take longer because the AI models need to download.
