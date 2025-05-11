import os
from groq import AsyncGroq
import asyncio
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL")

groq_client = None

async def init_groq_client():
    global groq_client
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set in environment variables")
    try:
        groq_client = AsyncGroq(api_key=GROQ_API_KEY)
        print("Successfully initialized AsyncGroq client.")
        return groq_client
    except Exception as e:
        raise ValueError(f"Failed to initialize AsyncGroq client: {e}")

async def get_groq_streaming_response(client: AsyncGroq, context: str, query: str):
    """
    Gets a streaming response from Groq's chat completion using AsyncGroq.
    Yields content chunks (tokens) as they are received.
    """
    if not client:
        raise ValueError("Groq client not initialized.")

    prompt = f"""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Do not make assumptions or provide information outside of the given context.    

Question: {query}
Context: {context}

Answer:"""

    try:
        stream = await client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=GROQ_MODEL_NAME,
            stream=True,
        )
        print(f"Streaming response from Groq: {stream}")
        async for chunk in stream:
            if chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        error_detail = str(e)
        if hasattr(e, 'body') and e.body:
            error_detail = f"{error_detail} - API Response: {e.body}"
        print(f"Error during Groq API call: {error_detail}")
        yield f"Error from LLM: {error_detail}" 

