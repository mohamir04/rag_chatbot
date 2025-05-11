import os
from pinecone import Pinecone, ServerlessSpec

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX")
PINECONE_MODEL_NAME = "multilingual-e5-large"
PINECONE_DIMENSION = 1024  # Dimension for multilingual-e5-large
PINECONE_METRIC = "cosine"   # Common metric for sentence embeddings
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

pinecone_client = None
index = None

def init_pinecone_index():
    global pinecone_client, index
    if not PINECONE_API_KEY:
        print("Error: PINECONE_API_KEY is not set.")
        raise ValueError("PINECONE_API_KEY is not set in environment variables.")

    try:
        pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
        
        # Get existing index names
        indexes_on_server = pinecone_client.list_indexes()
        existing_index_names = [index.name for index in indexes_on_server] # ListResponse is iterable

        if PINECONE_INDEX_NAME not in existing_index_names:
            print(f"Index '{PINECONE_INDEX_NAME}' not found. Creating a new one with model integration...")
            pinecone_client.create_index_for_model(
                name=PINECONE_INDEX_NAME,
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION,
                embed={
                    "model": PINECONE_MODEL_NAME,
                    "field_map": {
                        "text": "text" 
                    }
                }
            )
            print(f"Index '{PINECONE_INDEX_NAME}' created successfully with model '{PINECONE_MODEL_NAME}'.")
        else:
            print(f"Index '{PINECONE_INDEX_NAME}' already exists.")

        index = pinecone_client.Index(PINECONE_INDEX_NAME)
        print(f"Successfully connected to Pinecone index: {PINECONE_INDEX_NAME}")
        print(index.describe_index_stats())
        return index
    except Exception as e:
        raise ValueError(f"Pinecone initialization failed: {str(e)}")

def upsert_documents(pinecone_index, documents: list[dict], namespace: str = ""):
    """
    Upserts documents to the Pinecone index.
    Input documents: {"id": "...", "text": "...", "source": "...", "page_number": ...}
    All these fields (id, text, source, page_number) are stored as top-level attributes.
    Pinecone embeds the content of the "text" field due to field_map.
    """
    if not pinecone_index:
        raise ValueError("Pinecone index not initialized.")
    if not documents:
        print("No documents provided for upserting.")
        return {"upserted_count": 0, "message": "No documents provided."}

    records_to_upsert = []
    for doc in documents:
        if not all(k in doc for k in ["id", "text", "source", "page_number"]):
            print(f"Skipping doc due to missing required keys (id, text, source, page_number): {doc.get('id', 'N/A')}")
            continue
        if not doc["text"].strip():
            print(f"Skipping doc due to empty 'text': {doc.get('id', 'N/A')}")
            continue
        
        # All fields are now top-level
        records_to_upsert.append({
            "id": doc["id"],
            "text": doc["text"],
            "source": doc["source"],
            "page_number": doc["page_number"]
        })
    
    if not records_to_upsert:
        return {"message": "No valid documents after filtering."}

    try:
        pinecone_index.upsert_records(records=records_to_upsert, namespace=namespace)
        return {"message": "PDF processed and content stored."}
    except Exception as e:
        print(f"Error upserting to Pinecone: {e}")
        raise ValueError(f"Pinecone upsert failed: {str(e)}")

def retrieve_from_pinecone(pinecone_index, query_text: str, top_k: int = 3, namespace: str = ""):
    """
    Retrieves relevant documents from Pinecone.
    Pinecone, with create_index_for_model and field_map, returns user-supplied fields 
    (like source, page_number) in match.fields.
    """
    if not pinecone_index:
        raise ValueError("Pinecone index not initialized.")
    
    print(pinecone_index.describe_index_stats())
    
    try:
        search_payload = {"inputs": {"text": query_text}, "top_k": top_k}
        print(f"Attempting to query Pinecone with payload: {search_payload}, namespace: '{namespace if namespace else "default"}'")
        query_response = pinecone_index.search(
            query=search_payload, 
            namespace=namespace 
        )
        # print(f"Raw query_response from Pinecone: {query_response}")
        
        contexts = []
        hits = query_response.get('result', {}).get('hits')

        if hits and isinstance(hits, list):
            for hit_item in hits:
                if hit_item.get('_id') and hit_item.get('_score') and hit_item.get('fields'):
                    fields = hit_item['fields']
                    context_item = {
                        "id": hit_item['_id'],
                        "text": fields.get('text'),
                        "score": hit_item['_score'],
                        "source": fields.get("source"),
                        "page_number": fields.get("page_number")
                    }
                    contexts.append(context_item)
                else:
                    print(f"Warning: Skipping a hit due to unexpected structure or missing fields. Hit: {hit_item}")
        else:
            print(f"Warning: No 'hits' found in Pinecone response or 'hits' is not a list. Response: {query_response}")

        return contexts
    except Exception as e:
        raise ValueError(f"Pinecone query failed: {str(e)}") 