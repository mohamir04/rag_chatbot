# RAG API with FastAPI, Pinecone, and Groq

This project implements a Retrieval Augmented Generation (RAG) API using FastAPI for the web framework, Pinecone as the vector database, and Groq for Large Language Model (LLM) interactions. It allows users to upload PDF documents or fetch content from URLs, store their text content in Pinecone, and then chat with an LLM that uses the stored documents as context.

## Features

*   **PDF Upload**: Upload PDF files to extract text and store in Pinecone.
*   **URL Fetching**: Provide a URL to fetch its main textual content and store in Pinecone.
*   **Vector Storage**: Uses Pinecone with the `multilingual-e5-large` embedding model.
*   **LLM Interaction**: Leverages Groq for fast, streaming LLM responses.
*   **Contextual Chat**: Retrieves relevant document chunks from Pinecone to provide context to the LLM for answering queries.
*   **Web Interface**: A simple HTML and JavaScript UI to interact with the API (upload documents and chat).

## Prerequisites

*   Python 3.8+
*   Pip (Python package installer)
*   Access to:
    *   Pinecone API (get an API key and create an index)
    *   Groq API (get an API key)

## Project Structure

```
.
├── .env.example        # Example environment variables
├── main.py             # FastAPI application, API endpoints, lifespan management
├── groq_utils.py       # Groq client initialization and LLM streaming logic
├── pinecone_utils.py   # Pinecone client initialization, upsert, and retrieval logic
├── web_utils.py        # Utilities for fetching and parsing web content
├── requirements.txt    # Python package dependencies
├── static/             # Static files for the UI
│   ├── index.html      # Main HTML file for the UI
└── README.md           # This file
```

## Setup

1.  **Clone the repository** (if applicable):
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create and activate a virtual environment**:
    *   On macOS and Linux:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    *   On Windows:
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables**:
    *   Create a `.env` file by copying the example:
        ```bash
        cp .env.example .env
        ```
    *   Open the `.env` file and fill in your actual API keys and Pinecone index name:
        ```env
        PINECONE_API_KEY=your_pinecone_api_key_here
        PINECONE_INDEX=your_pinecone_index_name_here
        GROQ_API_KEY=your_groq_api_key_here
        ```

## Running the Application

Once the setup is complete, you can run the FastAPI application using Uvicorn:

```bash
python main.py
```

This will typically start the server on `http://localhost:8000`. You can access the UI by navigating to this address in your web browser.

Alternatively, for development with auto-reload:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

The application exposes the following main API endpoints:

*   `GET /`: Serves the main HTML UI.
*   `POST /upload-pdf/`:
    *   Accepts a PDF file (`multipart/form-data`).
    *   Extracts text page by page.
    *   Stores each page's content with `source` (filename) and `page_number` metadata in Pinecone.
*   `POST /upload-url/`:
    *   Accepts a JSON payload: `{"url": "your_url_here"}`.
    *   Fetches and parses the main text content from the URL.
    *   Stores the content with `source` (URL) and `page_number` (defaults to 1) metadata in Pinecone.
*   `POST /chat/`:
    *   Accepts a JSON payload: `{"query": "your_question_here", "top_k": 6}` (top_k is optional).
    *   Retrieves relevant document chunks from Pinecone based on the query.
    *   Streams a response from Groq, prefixed with JSON source information for the UI to display.

The chat stream format is:
```
{"sources": [{"source": "doc_name_or_url", "pages": [1,2], "is_url": false/true}, ...]}
###LLM_ANSWER###
LLM actual response stream...
```

## Notes

*   Ensure your Pinecone index is configured to use the `multilingual-e5-large` model (1024 dimensions).
*   The `web_utils.py` uses BeautifulSoup4 for parsing HTML content from URLs. The quality of text extraction can vary depending on the website structure. 